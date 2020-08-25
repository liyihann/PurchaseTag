import csv
import re
import jieba

tags_dict = {'数码':1, '珠宝':2, '玩具':3, '钟表':4, '汽车摩托':5, '健康运动、户外':6, '厨具':7, '医药':8, '宠物':9, '礼品':10, '食品饮料、保健食品':11, '家用电器':12, '手机':13, '电脑、软件、办公':14, '出版物':15, '服饰、鞋、包':16, '母婴童':17, '个护化妆':18, '家居家装':19}


def get_product_tag(file):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    products = list()
    tags = list()
    with open(file,'r') as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            product = line[2]
            tag = line[3]
            filtered = "".join(re.findall(pattern, product))
            segment = jieba.lcut(filtered)
            products.append(segment)
            tags.append(tags_dict.get(tag))
    return products,tags

def write_csv(path,list1,list2):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        for index in range(list1.__len__()):
            data_row = [list1[index],list2[index]]
            csv_write.writerow(data_row)

products,tags = get_product_tag('data/data.csv')

write_csv('data/data_preprocessed.csv',products,tags)


