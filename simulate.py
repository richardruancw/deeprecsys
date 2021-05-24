"""Script to generate recommendation data from simulation"""
import argparse
from argparse import Namespace
import os

import pandas as pd  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from scipy import sparse as sp  # type: ignore
from tqdm import tqdm  # type: ignore
from deeprecsys.data import RatingData
from deeprecsys.module import FactorModel, NoiseFactor
from deeprecsys.recommender import ClassRecommender, RatingEstimator
from sklearn.model_selection import train_test_split


def main(args: Namespace):
    ratings = pd.read_feather(os.path.join(args.data_path, args.data_name))
    u_num, i_num = ratings.uidx.max() + 1, ratings.iidx.max() + 1
    rel_factor = FactorModel(u_num, i_num, args.dim)
    expo_factor = FactorModel(u_num, i_num, args.dim)
    rating_features = list(zip(ratings.uidx, ratings.iidx, ratings.rating))
    rating_model = RatingEstimator(u_num, i_num, rel_factor)

    # expo_model = BPRRecommender(u_num, i_num, expo_factor)
    expo_model = ClassRecommender(u_num, i_num, expo_factor)

    #
    CUDA = None
    print('train rel model')
    rating_model.fit(rating_features, cuda=CUDA, num_epochs=args.epoch)

    #
    print('train expo model')
    expo_model.fit(ratings, cuda=CUDA, num_epochs=args.epoch, decay=args.decay)

    torch.save(rel_factor.state_dict(), os.path.join(args.data_path, f'rel.pt'))
    torch.save(expo_factor.state_dict(), os.path.join(args.data_path, f'expo.pt'))

    print('get noise added expo model')
    expo_factor = NoiseFactor(expo_factor, args.dim)
    # expo_factor = expo_factor.cuda()
    torch.save(expo_factor.state_dict(), os.path.join(args.data_path, f'expo_noise.pt'))
    # re-assign the expo model
    expo_model = ClassRecommender(u_num, i_num, expo_factor)

    sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))
    if not args.sample_sim:
        if u_num * i_num > 10000 * 10000:
            raise ValueError('Size over limit, please use --sample_sim flag')

        u_all = np.arange(u_num).repeat(i_num)
        i_all = np.arange(i_num).repeat(u_num).reshape(i_num, u_num).reshape(-1, order='F')
        est_rel = rating_model.score(u_all, i_all)
        est_click_prob = sigmoid(est_rel - args.epsilon)
        est_logits = expo_model.score(u_all, i_all)
        est_expo_prob = sigmoid(est_logits) ** args.p
        simu_size = len(est_click_prob)

        # expo_event = np.random.random(simu_size) < est_expo_prob

        # The item has to been exposed to then clicked by a customer
        est_expo_prob_mat = est_expo_prob.reshape(u_num, i_num)
        est_expo_threshold= np.sort(est_expo_prob_mat)[:, int(i_num * args.quantile)]
        est_expo_prob_mat = est_expo_prob_mat > est_expo_threshold[:, np.newaxis]
        expo_event = est_expo_prob_mat.flatten()

        click_event = np.random.random(simu_size) < est_click_prob

        valid = click_event * expo_event
        out = {}
        out['uidx'] = u_all
        out['iidx'] = i_all
        out['click_prob'] = est_click_prob
        out['expo_prob'] = est_expo_prob
        out['click'] = click_event * expo_event
        out['expo'] = expo_event
        out_df = pd.DataFrame(out)
        out_df.to_feather(os.path.join(args.data_path, f'sim_full.feather'))

        print(f'total size: {len(valid)}, valid size: {valid.sum()}')
        out = {}
        out['uidx'] = u_all[valid]
        out['iidx'] = i_all[valid]
        out['click_prob'] = est_click_prob[valid]
        out['expo_prob'] = est_expo_prob[valid]
        out_df = pd.DataFrame(out)
    else:
        print('Too many items to compute, only consider a subset')
        template = np.ones(args.item_sample_size).astype(np.int64)
        out = {'uidx': [], 'iidx': [], 'click_prob': [], 'expo_prob': []}
        for i in tqdm(range(u_num)):
            candidate_item = np.random.randint(low=0, high=i_num, size=args.item_sample_size)
            candidate_user = template * i

            est_rel = rating_model.score(candidate_user, candidate_item)
            est_click_prob = sigmoid(est_rel - args.epsilon)
            est_logits = expo_model.score(candidate_user, candidate_item)
            est_expo_prob = sigmoid(est_logits) ** args.p

            click_event = np.random.random(args.item_sample_size) < est_click_prob
            expo_event = np.random.random(args.item_sample_size) < est_expo_prob
            valid = click_event * expo_event

            if valid.sum() >= 1:
                out['uidx'].extend(candidate_user[valid].tolist())
                out['iidx'].extend(candidate_item[valid].tolist())
                out['click_prob'].extend(est_click_prob[valid].tolist())
                out['expo_prob'].extend(est_expo_prob[valid].tolist())
        if len(out['uidx']) == 0:
            raise ValueError('Simulation failed, does not gather positive signals')
        out_df = pd.DataFrame(out)

    out_df['ts'] = np.random.rand(out_df.shape[0])

    train_df, tmp_df = train_test_split(out_df, test_size=0.2)
    val_df, test_df = train_test_split(tmp_df, test_size=0.5)

    train_df = train_df.reset_index(drop=True)
    print(f'train shape: {train_df.shape}')
    val_df = val_df.reset_index(drop=True)
    print(f'val shape: {val_df.shape}')
    test_df = test_df.reset_index(drop=True)
    print(f'test shape: {test_df.shape}')
    print(train_df.head())

    train_df.to_feather(os.path.join(args.data_path, f'sim_train.feather'))
    val_df.to_feather(os.path.join(args.data_path, f'sim_val.feather'))
    test_df.to_feather(os.path.join(args.data_path, f'sim_test.feather'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--epsilon', type=float, default=4, help='Simulation parameters, refer to paper for details')
    parser.add_argument('--epoch', type=float, default=5, help='Simulation parameters, refer to paper for details')
    parser.add_argument('--p', type=float, default=3, help='Simulation parameters, refer to paper for details')
    parser.add_argument('--decay', type=float, default=1e-8, help='L2 penalty for embeddings')
    parser.add_argument('--quantile', type=float, default=0.9, help='Only items in top x percent quantile are exposed '
                                                                    'to users')
    parser.add_argument('--data_path', type=str, default='data/ml-1m/ml-1m')
    parser.add_argument('--data_name', type=str, default='ratings.feather')
    parser.add_argument('--sample_sim', action='store_true')
    parser.add_argument('--item_sample_size', type=int, default=2000)
    args = parser.parse_args()
    main(args)