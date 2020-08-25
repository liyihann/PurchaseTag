import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib
import os
import codecs
import shutil
import json


print('Loading data...')
train = pd.read_csv('data/train.csv', dtype=object)
validate = pd.read_csv('data/validate.csv', dtype=object)

print('Vectorizing...')
vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2), min_df=10, max_df=0.5)
corpus = train['product']
X = vectorizer.fit_transform(corpus.values)
y = train['categoryid'].values

# train
categoryid_set = set(train['categoryid'].values)
print('Training...')
clf = MultinomialNB()
chunk = 50000
m = X.shape[0]
if m < chunk:
    clf.fit(X, y)
else:
    for i, idx in enumerate(np.split(np.arange(m), range(chunk, m, chunk))):
        print('\tTraining %d chunk' % (i + 1))
        clf.partial_fit(X[idx], y[idx], classes=list(categoryid_set))

# validate
print('Cross validating...')
X_validate = vectorizer.transform(validate['product'])
y_true = validate['categoryid'].values
jll = clf.predict_proba(X_validate)  # joint likelihood
y_pred = clf.classes_[np.nanargmax(jll, axis=1)]
max_proba = np.nanmax(jll, axis=1)

if not os.path.isdir('output'):
    os.mkdir('output')

# trade off between acurry and recall
# search best decision boundry for each category
def search(persistence):
    print('Searching: ')
    boundary_of_category = dict()
    max_p_category = np.nanmax(jll, axis=0)  # max probability in each category
    min_p_category = np.nanmin(jll, axis=0)  # min probability in each category
    for categoryid in categoryid_set:
        print('Searching in %s' % (categoryid))
        idx = np.where(clf.classes_ == categoryid)
        tp = (y_true == categoryid) & (y_pred == categoryid)
        fp = (y_true != categoryid) & (y_pred == categoryid)
        fn = (y_true == categoryid) & (y_pred != categoryid)
        proba_tp = np.sort(max_proba[tp])
        proba_fp = np.sort(max_proba[fp])
        proba_fn = np.sort(max_proba[fn])
        threshold = np.linspace(min_p_category[idx], max_p_category[idx], 100)
        tp_num = proba_tp.shape[0] - np.searchsorted(proba_tp, threshold)
        fp_num = proba_fp.shape[0] - np.searchsorted(proba_fp, threshold)
        fn_num = proba_fn.shape[0] + np.searchsorted(proba_tp, threshold)
        accuracy = np.true_divide(tp_num, (tp_num + fp_num))
        recall = np.true_divide(tp_num, (tp_num + fn_num))
        f1 = np.true_divide(2 * accuracy * recall, (accuracy + recall))
        idx_max_f1 = np.nanargmax(f1)
        boundary_of_category[categoryid] = threshold[idx_max_f1].tolist()
        y_pred[(max_proba < threshold[idx_max_f1])
               & (y_pred == categoryid)] = None
    if persistence:
        with codecs.open('output/boundary.json', encoding='utf-8', mode='w') as f:
            json.dump(obj=boundary_of_category, fp=f, ensure_ascii=False, indent=4, separators=(',', ': '))

search(True)


with open('output/report.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_pred), file=f)

# model persistence in binary
if not os.path.isdir('bin'):
    os.mkdir('bin')
joblib.dump(vectorizer, 'bin/tfidf')
joblib.dump(clf, 'bin/classifier')

# Output model in a readable format
persistence = True
if persistence:
    output_dir = 'output/log_proba'
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)
    words = vectorizer.get_feature_names()
    for i in range(len(clf.feature_log_prob_)):
        pairs = []
        log_proba = clf.feature_log_prob_[i]
        class_id = clf.classes_[i]
        for j in range(len(log_proba)):
            word = words[j]
            pairs.append((word, log_proba[j]))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        output_path = os.path.join(output_dir, '%s.txt' % class_id)
        with codecs.open(output_path, encoding='utf-8', mode='w') as f:
            for word, weight in pairs:
                print('%s:%f' % (word, weight), file=f)

print('Finish')