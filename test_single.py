import re
import jieba
import joblib
import numpy as np
import json

category_dict = {'1':'数码', '2':'珠宝', '3':'玩具', '4':'钟表', '5':'汽车摩托', '6':'健康运动、户外', '7':'厨具', '8':'医药', '9':'宠物', '10':'礼品', '11':'食品饮料、保健食品', '12':'家用电器', '13':'手机', '14':'电脑、软件、办公', '15':'出版物', '16':'服饰、鞋、包', '17':'母婴童', '18':'个护化妆', '19':'家居家装'}

def preprocess(text):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    filtered = "".join(re.findall(pattern, text))
    segmented = u' '.join(jieba.lcut(filtered))
    return segmented

def filter(predicted_cateid,probability):
    with open('bin/boundary.json') as f:
        boundary = json.load(f)
    if probability >= boundary[predicted_cateid]:
        return predicted_cateid
    else:
        return None

product = "惠普HP Deskjet 1050 J410a彩色喷墨一体机(打印 复印 扫描)"


preprocessed = preprocess(product)
print(preprocessed)

tfidf = joblib.load('bin/tfidf')
clf = joblib.load('bin/classifier')

X = tfidf.transform([product])
print(X)
jll = clf.predict_proba(X)  # joint likelihood
print(jll)
y_pred = clf.classes_[np.nanargmax(jll, axis=1)] # predicted_categoryid
print(y_pred)
max_proba = np.nanmax(jll, axis=1)
print(max_proba)

# predicted_categoryid = filter(y_pred.__getitem__(0),max_proba)
# predicted_category = category_dict.__getitem__(predicted_categoryid)
#
#
# print(predicted_category)