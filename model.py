import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from datetime import datetime
import joblib
import os
import codecs
import shutil


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
        print('\t%s\tTraining %d chunk' % (datetime.now(), (i + 1)))
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
with open('output/report.txt', 'w') as f:
    print(metrics.classification_report(y_true, y_pred), file=f)

# model persistence in binary
if not os.path.isdir('bin'):
    os.mkdir('bin')
joblib.dump(vectorizer, 'bin/vectorizer')
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