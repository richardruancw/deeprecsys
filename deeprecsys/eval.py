import logging
from typing import List, Optional, Dict, Set

import pandas as pd
from numpy.random import RandomState
from scipy import sparse as sp #type: ignore
import numpy as np #type: ignore
from sklearn.utils.extmath import randomized_svd #type: ignore
from tqdm import tqdm #type: ignore
from deeprecsys.base_recommender import Recommender


def unbiased_eval(num_user: int, num_item: int, dat_df: pd.DataFrame,
                  recom: Recommender, rel_model: Optional[Recommender] = None,
                  expo_model: Optional[Recommender] = None,
                  past_hist: Optional[Dict[int, Set[int]]] = None, expo_compound: float = 1.0,
                  epsilon: float = 1.0, num_neg: int = 100, cut_len: int = 10, seed: int = 886):
    logger = logging.getLogger(__name__)
    # this is to make sure comparision between models is fair yet not affect the negative sampling's variation
    prng = RandomState(seed)

    row, col = dat_df.uidx, dat_df.iidx
    def sigmoid(x): return np.exp(x) / (1 + np.exp(x))
    recall_cnt = 0
    ndcg_sum = 0
    for u, i in list(zip(row, col)):
        if past_hist is None:
            neg = prng.randint(0, num_item, num_neg)
            neg = neg[neg != i]
        else:
            neg = prng.randint(0, num_item, num_neg)
            for idx in range(num_neg):
                if int(neg[idx]) in past_hist.get(u, []) or i == neg[idx]:
                    while int(
                            neg[idx]) not in past_hist.get(
                            u, []) and i != neg[idx]:
                        neg[idx] = prng.randint(0, num_item)
        item_list: List[int] = neg.tolist()
        item_list.append(i)
        user_list = [u] * len(item_list)
        scores = recom.score(user_list, item_list)
        if rel_model is not None:
            rel_score = rel_model.score(user_list, item_list)
            rel_prob = sigmoid(rel_score - epsilon)
        else:
            rel_prob = np.ones(len(scores))

        expo_score = 1
        if expo_model is not None:
            expo_score = sigmoid(expo_model.score([u], [i])[0]) ** expo_compound

        rank = scores.argsort()[::-1]
        item_npy = np.array(item_list)
        top_items = item_npy[rank][:cut_len]
        top_item_rel_prob = rel_prob[rank][:cut_len]
        #recall_cnt += int(i in top_items)
        for pos, (top_i, top_rel) in enumerate(
                zip(top_items, top_item_rel_prob)):
            if i == top_i:
                recall_cnt += (top_rel / expo_score)
                ndcg_sum += np.log(2) / np.log(2 + pos) * \
                    (top_rel / expo_score)
    logger.info(
        f'Recall@{cut_len} = {recall_cnt / len(row):.5f}; NDCG@{cut_len} = {ndcg_sum / len(row):.5f}')
    return {'recall': recall_cnt / len(row), 'ndcg': ndcg_sum / len(row)}



