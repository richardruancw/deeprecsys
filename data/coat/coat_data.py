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
te_df = extract(test_rating.data)

tr_df.to_feather('./coat_data/train.feather')
te_df.to_feather('./coat_data/val.feather')
te_df.to_feather('./coat_data/test.feather')

df = pd.concat([tr_df, te_df])
df = df.reset_index(drop=True)
df.to_feather('./coat_data/ratings.feather')