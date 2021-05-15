from typing import List
import os
import time
import argparse
from argparse import Namespace
import logging

from scipy import sparse as sp  # type: ignore
import numpy as np  # type: ignore
from sklearn.utils.extmath import randomized_svd  # type: ignore
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse as sp  # type: ignore
import torch  # type: ignore

from deeprecsys.module import *
from deeprecsys.recommender import *
from deeprecsys.eval import unbiased_eval
from deeprecsys import data, reweight

POSITIVE_RATING_THRESHOLD = 3


def frame2mat(df, num_u, num_i):
    row, col = df.uidx, df.iidx
    data = np.ones(len(row))
    mat = sp.csr_matrix((data, (row, col)), shape=(num_u, num_i))
    return mat


def main(args: Namespace):
    ratings = pd.read_feather(os.path.join(args.data_path, args.data_name))
    user_num, item_num = ratings.uidx.max() + 1, ratings.iidx.max() + 1

    tr_df = pd.read_feather(os.path.join(args.data_path, 'train.feather'))
    val_df = pd.read_feather(os.path.join(args.data_path, 'val.feather'))
    te_df = pd.read_feather(os.path.join(args.data_path, 'test.feather'))

    if not args.tune_mode:
        tr_df = pd.concat([tr_df, val_df])
        te_df = te_df
    else:
        tr_df = tr_df
        te_df = val_df

    # Rating >= 3 is positive, otherwise it is negative
    tr_df['rating'] = tr_df['rating'] > POSITIVE_RATING_THRESHOLD
    te_df['rating'] = te_df['rating'] > POSITIVE_RATING_THRESHOLD

    # Create user history features
    past_hist = tr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
    item_cnt_dict = tr_df.groupby('iidx').count().uidx.to_dict()
    item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(item_num)])
    labeled_hist = tr_df.groupby('uidx').apply(
        lambda x: list(zip(x.ts, x.iidx, x.rating))).to_dict()
    for k in labeled_hist.keys():
        labeled_hist[k] = [(x[1], x[2]) for x in sorted(labeled_hist[k])]

    logger.info(f'test data size: {te_df.shape}')

    f_module = FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
    w_module = FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
    g_module = FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)

    print(isinstance(f_module, SeqModelMixin))

    # TODO: Add reweight model training on real data using both sparse and seq model. It needs to converge

    rw_m = reweight.ReWeightLearner(f=f_module, g=g_module, w=w_module,
                                    lambda_=args.lambda_, user_num=user_num, item_num=item_num)

    rw_m.fit(tr_df=tr_df, test_df=te_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--decay', type=float, default=1e-7)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data/ml-1m/ml-1m')
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--prefix', type=str, default='ml_1m_real')
    parser.add_argument('--num_neg', type=str, default=4)
    parser.add_argument('--tune_mode', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_len', type=int, default=50)

    args = parser.parse_args()

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'log/{args.prefix}-{str(time.time())}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    main(args)
