import logging
from typing import Optional, Dict, Set

import numpy as np
import pandas as pd
import torch

from deeprecsys import data
from deeprecsys.data import NegSeqData, NegSequenceData
from deeprecsys.recommender import Recommender, ClassRecommender, build_optimizer, unbiased_eval, DeepRecommender


def ac_train_v2(f_model: torch.nn.Module,
                g_model: torch.nn.Module,
                beta_model: torch.nn.Module,
                tr_df: pd.DataFrame,
                user_num: int,
                item_num: int,
                val_df: Optional[pd.DataFrame] = None,
                rating_model: Optional[Recommender] = None,
                expo_model: Optional[Recommender] = None,
                past_hist: Optional[Dict[int, Set[int]]] = None,
                num_epochs: int = 50,
                batch_size: int = 2048,
                min_prob: float = 0.1,
                num_neg: int = 1,
                cuda_idx: int = 0,
                min_delta: float = 0.1,
                lr: float = 0.01,
                f_round_ahead: int = 1,
                g_round_ahead: int = 1,
                decay: float = 0.0):
    logger = logging.getLogger(__name__)
    with torch.cuda.device(cuda_idx):

        f_recommender = ClassRecommender(user_num, item_num, f_model)
        g_recommender = ClassRecommender(user_num, item_num, g_model)

        u, v = tr_df.uidx.tolist(), tr_df.iidx.tolist()

        minimizer = build_optimizer(lr, f_model, beta_model)
        maximizer = build_optimizer(lr, g_model)

        loss_func = torch.nn.BCELoss(reduction='none')
        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        #device_cuda = torch.device(f'cuda:{cuda_idx}')
        f_model.cuda()
        g_model.cuda()
        beta_model.cuda()

        def train_epoch(optimizer, data_loader, flag='g_train'):
            f_loss_record, g_loss_record = [], []
            # train the g_model for one epoch
            for c_round in range(g_round_ahead):
                for user, item_pos, item_neg_list in data_loader:

                    f_model.zero_grad()
                    g_model.zero_grad()
                    beta_model.zero_grad()
                    optimizer.zero_grad()

                    f_model.train()
                    g_model.train()
                    beta_model.train()

                    user = user.long().cuda()
                    item_pos = item_pos.long().cuda()
                    item_neg_list = item_neg_list.cuda().long()
                    item_neg = item_neg_list.flatten()

                    user_for_neg = user.reshape(
                        1, -1).repeat(num_neg, 1).t().flatten()
                    user = torch.cat([user, user_for_neg], dim=0).long()
                    items = torch.cat([item_pos, item_neg], dim=0).long()
                    labels = torch.cat([torch.ones(len(item_pos)).cuda(
                    ), torch.zeros(len(item_neg)).cuda()], dim=0).float()

                    f_s = f_model(user, items)
                    g_s = g_model(user, items)
                    q_s = beta_model(user, items, g_s, labels)

                    f_prob = torch.clamp(act_func(f_s), min=0.01, max=1)
                    g_prob = torch.clamp(act_func(g_s), min=0.01, max=1)
                    q_prob = torch.clamp(act_func(q_s), min=min_prob, max=1)

                    f_loss = -1 * (labels * torch.log(f_prob) +
                                   (1 - labels) * torch.log(1 - f_prob)) / q_prob
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob))

                    if flag == 'g_train':
                        target = (
                            torch.clamp(
                                min_delta + g_loss - f_loss,
                                min=0)).mean()  # g wants to maximize the gap
                        target += decay * g_model.get_l2(user, items)
                        target.backward()
                    elif flag == 'f_train':
                        target = f_loss.mean()
                        target += decay * \
                            f_model.get_l2(user, items) + decay * \
                            beta_model.get_l2(user, items)
                        target.backward()
                    else:
                        raise ValueError('use g_train or f_train')
                    optimizer.step()

                    with torch.no_grad():
                        f_loss = f_loss.mean()
                        g_loss = g_loss.mean()
                        f_loss_record.append(f_loss.item())
                        g_loss_record.append(g_loss.item())

                logger.info(
                    f'{flag} at {c_round} round -- f_loss: {np.mean(f_loss_record)} g_loss: {np.mean(g_loss_record)}')

        # pre-fit the g without adjusting
        g_recommender.fit(tr_df,
                          num_epochs=0,
                          cuda=cuda_idx,
                          decay=decay)

        neg_data = NegSeqData(list(zip(u, v)), item_num,
                              num_neg=num_neg, past_hist=past_hist)
        neg_data.is_training = True
        for epoch in range(num_epochs):
            neg_data.ng_sample()
            data_loader = data.DataLoader(
                neg_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True)

            logger.info(f'Epoch -- {epoch}')
            minimizer.zero_grad()
            maximizer.zero_grad()

            train_epoch(minimizer, data_loader, 'f_train')
            train_epoch(maximizer, data_loader, 'g_train')

            if val_df is not None:
                logger.info('f_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    f_recommender,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist)

                logger.info('g_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    g_recommender,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist)


def ac_train_v3(f_model: torch.nn.Module,
                is_f_seq: bool,
                g_model: torch.nn.Module,
                is_g_seq: bool,
                beta_model: torch.nn.Module,
                tr_df: pd.DataFrame,
                user_num: int,
                item_num: int,
                val_df: Optional[pd.DataFrame] = None,
                rating_model: Optional[Recommender] = None,
                expo_model: Optional[Recommender] = None,
                past_hist: Optional[Dict[int, Set[int]]] = None,
                g_weight: float = 1.0,
                num_epochs: int = 50,
                batch_size: int = 2048,
                min_prob: float = 0.1,
                num_neg: int = 1,
                cuda_idx: int = 0,
                min_delta: float = 0.1,
                lr: float = 0.01,
                decay: float = 0.0,
                expo_compound: float = 1.0,
                epsilon: float = 1.0):
    logger = logging.getLogger(__name__)
    with torch.cuda.device(cuda_idx):

        if is_f_seq:
            f_recommender = DeepRecommender(user_num, item_num, f_model)
        else:
            f_recommender = ClassRecommender(user_num, item_num, f_model)

        if is_g_seq:
            g_recommender = DeepRecommender(user_num, item_num, g_model)
        else:
            g_recommender = ClassRecommender(user_num, item_num, g_model)

        minimizer = build_optimizer(lr, f_model, beta_model)
        maximizer = build_optimizer(lr, g_model)

        loss_func = torch.nn.BCELoss(reduction='none')
        def act_func(x): return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        #device_cuda = torch.device(f'cuda:{cuda_idx}')
        f_model.cuda()
        g_model.cuda()
        beta_model.cuda()

        def train_epoch(optimizer, data_loader, flag, is_f_seq, is_g_seq, round_repeat=1):
            f_loss_record, g_loss_record = [], []
            q_prob_record = []
            # train the g_model for one epoch
            for c_round in range(round_repeat):
                for user, item_i, item_j_list, item_hist in data_loader:

                    f_model.zero_grad()
                    g_model.zero_grad()
                    beta_model.zero_grad()
                    optimizer.zero_grad()

                    f_model.train()
                    g_model.train()
                    beta_model.train()

                    # transfer to gpu
                    bsz = item_hist.shape[0]
                    user = user.cuda().long()  # [B]
                    item_i = item_i.cuda().long()  # [B]
                    item_j_list = item_j_list.cuda().long()  # [B, num_neg]
                    item_hist = item_hist.cuda().long()  # [B, max_len]

                    # reshape
                    item_i = item_i.view(-1, 1)  # [B, 1]
                    items = torch.cat([item_i, item_j_list],
                                      dim=1)  # [B, 1 + num_neg]
                    labels = (torch.arange(1 + num_neg).cuda()
                              < 1).float().repeat(bsz).view(bsz, -1)  # [B, 1 + num_neg]
                    users = user.unsqueeze(1).repeat(
                        1, 1 + num_neg)  # [B, 1 + num_neg]

                    f_s = f_model(items, item_hist) if is_f_seq else f_model(
                        users, items)
                    g_s = g_model(items, item_hist) if is_g_seq else g_model(
                        users, items)
                    q_s = beta_model(users, items, g_s, labels)

                    f_prob = torch.clamp(act_func(f_s), min=0.01, max=1)
                    g_prob = torch.clamp(act_func(g_s), min=0.01, max=1)
                    q_prob = torch.clamp(act_func(q_s), min=min_prob, max=1)

                    f_loss = -1 * (labels * torch.log(f_prob) +
                                   (1 - labels) * torch.log(1 - f_prob)) / q_prob
                    g_loss = -1 * (labels * torch.log(g_prob) +
                                   (1 - labels) * torch.log(1 - g_prob))

                    if flag == 'g_train':
                        target = (
                            torch.clamp(
                                min_delta + g_weight * g_loss - f_loss,
                                min=0)).mean()  # g wants to maximize the gap
                        target += decay * g_model.get_l2(user, items)
                        target.backward()
                    elif flag == 'f_train':
                        target = f_loss.mean()
                        target += decay * \
                            f_model.get_l2(user, items) + decay * \
                            beta_model.get_l2(user, items)
                        target.backward()
                    else:
                        raise ValueError('use g_train or f_train')
                    optimizer.step()

                    with torch.no_grad():
                        f_loss = f_loss.mean()
                        g_loss = g_loss.mean()
                        f_loss_record.append(f_loss.item())
                        g_loss_record.append(g_loss.item())
                        q_prob_record.append(q_prob.mean().item())

                logger.info(
                    f'{flag} at {c_round} round -- f_loss: {np.mean(f_loss_record)} g_loss: {np.mean(g_loss_record)}, q_prob: {np.mean(q_prob_record)}')

        hist = tr_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx))).to_dict()
        for k in hist.keys():
            hist[k] = [x[1] for x in sorted(hist[k])]
        if is_f_seq:
            f_recommender.set_user_record(hist)
        if is_g_seq:
            g_recommender.set_user_record(hist)

        padding_idx = item_num + 1
        max_len = 1
        if is_f_seq:
            max_len = f_model.max_len
        elif is_g_seq:
            max_len = g_model.max_len
        f_seq_data = NegSequenceData(
            hist,
            max_len,
            item_num=item_num,
            padding_idx=padding_idx,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=not is_f_seq)

        f_train_loader = data.DataLoader(
            f_seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        g_seq_data = NegSequenceData(
            hist,
            max_len,
            item_num=item_num,
            padding_idx=padding_idx,
            num_neg=num_neg,
            window=True,
            past_hist=past_hist,
            allow_empty=not is_g_seq)

        g_train_loader = data.DataLoader(
            g_seq_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        for epoch in range(num_epochs):
            logger.info(f'Epoch -- {epoch}')
            minimizer.zero_grad()
            maximizer.zero_grad()

            train_epoch(minimizer, f_train_loader,
                        'f_train', is_f_seq, is_g_seq)
            train_epoch(maximizer, g_train_loader,
                        'g_train', is_f_seq, is_g_seq)

            logger.info(f'beta_model: {beta_model.alpha.item()}, {beta_model.beta.item()}, {beta_model.label_coef.item()}')
            if val_df is not None:
                logger.info('f_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    f_recommender,
                    epsilon=epsilon,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist,
                    expo_compound=expo_compound)

                logger.info('g_model:')
                logger.info('--unbiased--')
                unbiased_eval(
                    user_num,
                    item_num,
                    val_df,
                    g_recommender,
                    epsilon=epsilon,
                    rel_model=rating_model,
                    expo_model=expo_model,
                    past_hist=past_hist,
                    expo_compound=expo_compound)