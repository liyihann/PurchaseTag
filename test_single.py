import re
import jieba
import joblib
import numpy as np
import json

category_dict = {'1':'数码', '2':'珠宝', '3':'玩具', '4':'钟表', '5':'汽车摩托', '6':'健康运动、户外', '7':'厨具', '8':'医药', '9':'宠物', '10':'礼品', '11':'食品饮料、保健食品', '12':'家用电器', '13':'手机', '14':'电脑、软件、办公', '15':'出版物', '16':'服饰、鞋、包', '17':'母婴童', '18':'个护化妆', '19':'家居家装'}

def preprocess(text):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    filtered = "".join(re.findall(pattern, np.unicode(text)))
    segmented = u' '.join(jieba.lcut(filtered))
    return segmented

def filter(predicted_cateid,probability):
    with open('bin/boundary.json') as f:
        boundary = json.load(f)
    if probability >= boundary[predicted_cateid]:
        return predicted_cateid
    else:
        return None

product = "PAWCARES 柏可心 口腔护理套餐A 小型宠物狗犬用 去除口腔异味4"


preprocessed = preprocess(product)

tfidf = joblib.load('bin/tfidf')
clf = joblib.load('bin/classifier')

X = tfidf.transform([preprocessed])

jll = clf.predict_proba(X)  # joint likelihood

y_pred = clf.classes_[np.nanargmax(jll, axis=1)] # predicted_categoryid

max_proba = np.nanmax(jll, axis=1)

predicted_categoryid = filter(y_pred.__getitem__(0),max_proba)

if predicted_categoryid!=None:
    predicted_category = category_dict.__getitem__(predicted_categoryid)
    print(predicted_category)