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

from deeprecsys import module, recommender
from deeprecsys import data, reweight
from deeprecsys.eval import unbiased_eval

POSITIVE_RATING_THRESHOLD = 0
ALLOWED_MODELS = ['mf', 'mlp', 'seq']


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
    #
    # te_df = te_df[te_df['rating'] > 0]

    logger.info(f'test data size: {te_df.shape}')

    past_hist = tr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
    item_cnt_dict = tr_df.groupby('iidx').count().uidx.to_dict()
    item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(item_num)])

    logger.info(f'test data size: {te_df.shape}')

    tr_mat = frame2mat(tr_df, user_num, item_num)
    pop_factor = module.PopularModel(item_cnt)
    logging.info('-------The Popularity model-------')
    pop_model = recommender.PopRecommender(pop_factor)
    logger.info('biased eval for plian popular model on test')
    unbiased_eval(user_num, item_num, te_df, pop_model, past_hist=past_hist)

    logger.info('-------The SVD model---------')
    sv = recommender.SVDRecommender(tr_mat.shape[0], tr_mat.shape[1], args.dim)
    logger.info(f'model with dimension {args.dim}')
    sv.fit(tr_mat)
    logger.info('biased eval for SVD model on test')
    unbiased_eval(user_num, item_num, te_df, sv, past_hist=past_hist)

    # logger.info('------Regular MF model ------')
    # mf_m = module.FactorModel(user_num, item_num, args.dim)
    # mf_recom = recommender.ClassRecommender(user_num, item_num, mf_m)
    # mf_recom.fit(tr_df, test_df=te_df,
    #                num_epochs=args.epoch,
    #                cuda=args.cuda_idx,
    #                decay=args.decay,
    #                num_neg=args.num_neg,
    #                batch_size=args.batch_size,
    #                past_hist=past_hist,
    #                lr=args.lr)
    # unbiased_eval(user_num, item_num, te_df, mf_recom, past_hist=past_hist)

    logger.info('------Reweight and rebalance model ------')
    if args.model == 'mf':
        f_module = module.FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
        w_module = module.FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
        g_module = module.FactorModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
    elif args.model == 'mlp':
        f_module = module.MLPRecModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
        w_module = module.MLPRecModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
        g_module = module.MLPRecModel(user_num=user_num, item_num=item_num, factor_num=args.dim)
    elif args.model == 'seq':
        f_module = module.AttentionModel(user_num, item_num, args.dim, max_len=args.max_len)
        g_module = module.AttentionModel(user_num, item_num, args.dim, max_len=args.max_len)
        w_module = module.AttentionModel(user_num, item_num, args.dim, max_len=args.max_len)
    else:
        raise ValueError(f'model not defined! choices are: {ALLOWED_MODELS}')

    rw_m = reweight.ReWeightLearner(f=f_module, g=g_module, w=w_module,
                                    lambda_=args.lambda_, user_num=user_num, item_num=item_num,
                                    w_lower_bound=args.w_lower_bound)

    rw_m.fit(tr_df=tr_df, test_df=te_df, decay=args.decay, max_len=args.max_len, cuda=args.cuda_idx,
             max_count=args.max_step, min_count=args.min_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=32, help='Dimension of the embedding')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--decay', type=float, default=1e-7, help='l2 regularization strength for sparse model')
    parser.add_argument('--cuda_idx', type=int, default=None, help='Which GPU to use, default is to use CPU')
    parser.add_argument('--data_path', type=str, default='data/ml-1m/ml-1m')
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--lambda_', type=float, default=0.1)
    parser.add_argument('--prefix', type=str, default='ml_1m_real')
    parser.add_argument('--tune_mode', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of sequence')
    parser.add_argument('--num_neg', type=str, default=3, help='Number of random negative samples per real label')
    parser.add_argument('--w_lower_bound', type=float, default=None, help='Lower bound of w(u, i), set it 1 will '
                                                                          'disable reweighitng')
    parser.add_argument('--model', type=str, default='mf', choices=ALLOWED_MODELS, help='Base model used in min-max '
                                                                                        'training')
    parser.add_argument('--max_step', type=int, default=1, help='number of batches per maximization step')
    parser.add_argument('--min_step', type=int, default=1, help='number of batches per minimization step')

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
