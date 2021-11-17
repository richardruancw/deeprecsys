"""Download from https://webscope.sandbox.yahoo.com/"""
import pandas as pd
import numpy as np

def load_ratings(filename):
    df = pd.read_csv(filename, sep='\t', header=None)
    df.columns = ['uidx', 'iidx', 'rating']
    df['ts'] = np.zeros_like(df['rating'])
    return df


tr_df = load_ratings('YahooR3_small/ydata-ymusic-rating-study-v1_0-train.txt')
sample_ind = np.random.random(size=tr_df.shape[0]) < 0.8
validation_df = tr_df[~sample_ind]
train_df = tr_df[sample_ind]
test_df = load_ratings('YahooR3_small/ydata-ymusic-rating-study-v1_0-test.txt')

train_df = train_df.reset_index(drop=True)
train_df.to_feather('./YahooR3_small/train.feather')

validation_df = validation_df.reset_index(drop=True)
validation_df.to_feather('./YahooR3_small/val.feather')
test_df.to_feather('./YahooR3_small/test.feather')

df = pd.concat([tr_df, test_df])
df = df.reset_index(drop=True)
df.to_feather('./YahooR3_small/ratings.feather')
