"""Firstly download data from https://www.cs.cornell.edu/~schnabts/mnar/ """

import numpy as np
import pandas as pd

def load_ratings(filename):
    raw_matrix = np.loadtxt(filename)
    return np.ma.array(raw_matrix, dtype=np.int, copy=False,
                          mask=raw_matrix <= 0, fill_value=0, hard_mask=True)


def load_propensities(filename):
    return np.loadtxt(filename)

def extract(data):
    x, y = data.nonzero()
    rating = np.zeros_like(x)
    for i, (ix, iy) in enumerate(zip(x, y)):
        rating[i] = data[ix, iy]
    ts = np.zeros_like(x)
    return pd.DataFrame({
        'uidx': x,
        'iidx': y,
        'rating': rating,
        'ts': ts
    })

train_rating = load_ratings('coat_data/train.ascii')
test_rating = load_ratings('coat_data/test.ascii')

tr_df = extract(train_rating.data)
sample_ind = np.random.random(size=tr_df.shape[0]) < 0.8
validation_df = tr_df[~sample_ind]
train_df = tr_df[sample_ind]

test_df = extract(test_rating.data)

train_df = train_df.reset_index(drop=True)
train_df.to_feather('./coat_data/train.feather')

validation_df = validation_df.reset_index(drop=True)
validation_df.to_feather('./coat_data/val.feather')
test_df.to_feather('./coat_data/test.feather')

df = pd.concat([tr_df, test_df])
df = df.reset_index(drop=True)
df.to_feather('./coat_data/ratings.feather')