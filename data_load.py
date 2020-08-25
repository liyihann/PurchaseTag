import pandas as pd
import numpy as np

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