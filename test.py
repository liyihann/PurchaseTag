import pandas as pd
import numpy as np
import jieba
import re
import joblib
import json
import os

category_dict = {'1':'数码', '2':'珠宝', '3':'玩具', '4':'钟表', '5':'汽车摩托', '6':'健康运动、户外', '7':'厨具', '8':'医药', '9':'宠物', '10':'礼品', '11':'食品饮料、保健食品', '12':'家用电器', '13':'手机', '14':'电脑、软件、办公', '15':'出版物', '16':'服饰、鞋、包', '17':'母婴童', '18':'个护化妆', '19':'家居家装'}

def preprocess(cells):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    filtered = ["".join(re.findall(pattern, np.unicode(cell))) for cell in cells.values]
    segmented = [u' '.join(jieba.lcut(cell)) for cell in filtered]
    return u' '.join(segmented)

def filter(row):
    with open('output/boundary.json') as f:
        boundary = json.load(f)
    if row['proba'] >= boundary[row['categoryid']].__getitem__(0):
        return row['categoryid']
    else:
        return None

test = pd.read_csv('data/test.csv',names =['product'], encoding='utf-8')

print('Preprocessing...')
test.fillna('')
test['preprocessed'] = test[['product']].apply(preprocess, axis=1)


print('Predicting...')
tfidf = joblib.load('bin/tfidf')
clf = joblib.load('bin/classifier')

test['categoryid'] = test['product'].map(lambda x: '')
test['proba'] = test['product'].map(lambda x: .0)

step = 20000
for idx in np.arange(0, test.shape[0], step):
    X = tfidf.transform(test['preprocessed'].iloc[idx: idx + step].values)
    jll = clf.predict_proba(X)  # joint likelihood
    y_pred = clf.classes_[np.nanargmax(jll, axis=1)]
    max_proba = np.nanmax(jll, axis=1)
    test['categoryid'].iloc[idx: idx + step] = y_pred
    test['proba'].iloc[idx: idx + step] = max_proba

# filter by decision boundary
test['categoryid'] = test[['categoryid', 'proba']].apply(filter, axis=1)

test['categoryname'] = test['categoryid'].map(category_dict)

if not os.path.isdir('output'):
    os.mkdir('output')

print('Outputing...')
test[['categoryid', 'categoryname', 'product']].to_csv('output/result.csv', encoding='utf-8', index=False)