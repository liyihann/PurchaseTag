import pandas as pd
import numpy as np
from sklearn.utils import shuffle



def get_train_validate_data():
    df = pd.read_csv('data/data_preprocessed.csv', header=None, error_bad_lines=False,
                     dtype=object,
                     names=['product', 'categoryid'])
    df = df.dropna()
    # bootstrap sampling
    bag = []
    grouped = df.groupby(by=['categoryid'])
    for categoryid, group in grouped:
        group = group.reset_index(drop=True)
        sampled_group = np.random.choice(group.shape[0], size=1000)
        bag.append(group.iloc[sampled_group])
    resampled_data = pd.concat(bag)

    # split train and test dataset
    r = np.random.random_sample((len(resampled_data)))
    train = resampled_data.iloc[r >= 0.25]
    test = resampled_data.iloc[r < 0.25]
    train.to_csv('data/train.csv', index=False, encoding='utf-8')
    test.to_csv('data/validate.csv', index=False, encoding='utf-8')



def get_test_data():
    # test data
    df_test = pd.read_csv('data/data.csv', header=None, error_bad_lines=False,
                     dtype=object,
                     names=['n1','n2','product','category','n3','n4'])
    df_test = df_test.dropna()
    # bootstrap sampling
    bag = []
    grouped = df_test.groupby(by=['category'])
    for categoryid, group in grouped:
        group = group.reset_index(drop=True)
        sampled_group = np.random.choice(group.shape[0], size=200)
        bag.append(group.iloc[sampled_group])

    resampled_data = pd.concat(bag)[['category','product']]
    resampled_data = shuffle(resampled_data)
    resampled_data.to_csv('data/test.csv', index=False, encoding='utf-8',header=None)


# get_train_validate_data()

get_test_data()