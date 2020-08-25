import csv
import os
import pandas as pd

# 根据类别ID获取类别
def getFullCategory(catID):
    with open('data/allcategories.csv','r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == catID:
                cat = row[1]
    return cat

def combineFiles(file_dir):
    all = list()
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            with open(file_dir+"/"+file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    all.append(row)
    return all

def write_csv(path,data_list):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        for index in range(data_list.__len__()):
            if(len(data_list.__getitem__(index))>0):
                data_row = [data_list.__getitem__(index)[0], data_list.__getitem__(index)[1], data_list.__getitem__(index)[2], data_list.__getitem__(index)[1].split('|')[0], data_list.__getitem__(index)[1].split('|')[1], data_list.__getitem__(index)[1].split('|')[2]]
            csv_write.writerow(data_row)

def read_csv(path):
    with open(path,"r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            print(line)


# all_data = combineFiles("data/dataset")
# print(all_data.__len__())
# write_csv("data/data.csv",all_data)
