"""Implement the re-weight based learner"""
from bisect import bisect
from enum import Enum
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from deeprecsys import recommender
from deeprecsys.data import LabeledSequenceData
from deeprecsys.eval import unbiased_eval
from deeprecsys.module import SeqModelMixin, SparseModelMixin


class OptimizationStep(Enum):
    ARG_MIN = 0
    ARG_MAX = 1


def conditional_forward(user, item, user_hist, m: torch.nn.Module):
    """Sparse model use (user, item), seq model use (item, user_hist)
    """
    if isinstance(m, SeqModelMixin):
        return m(item, user_hist)
    elif isinstance(m, SparseModelMixin):
        return m(user, item)
    else:
        raise ValueError('Model type not supported')


def conditional_l2(user, item, m):
    """Sparse model use (user, item) to get active l2 penalty
    """
    if isinstance(m, SparseModelMixin):
        return m.get_l2(user, item)
    else:
        return 0


def aggregate_l2(user, item, *models):
    l2 = 0
    for m in models:
        l2 += conditional_l2(user, item, m)
    return l2


def list2deivce(vars, device):
    return [v.to(device) for v in vars]


class AlternatingCounter:
    def __init__(self, step_specs: Dict[Enum, int]):
        self.counter = 0
        curr_size = 0
        upper_bounds = []
        bound2enum = {}
        for k, step in step_specs.items():
            if step > 0:
                curr_size += step
                upper_bounds.append(curr_size)
                bound2enum[curr_size] = k
        self.max_size = curr_size
        self.uppper_bounds = upper_bounds
        self.bound2enum = bound2enum

    def touch(self):
        idx = bisect(self.uppper_bounds, self.counter)
        self.counter = (self.counter + 1) % self.max_size
        return self.bound2enum[self.uppper_bounds[idx]]


class ReWeightLearner:
    def __init__(self, f: torch.nn.Module,
                 w: torch.nn.Module,
                 g: torch.nn.Module,
                 item_num: int, user_num: int,
                 lambda_: float = 1,
                 w_lower_bound: Optional[float] = None) -> None:
        self.f = f
        self.w = w
        self.g = g
        self.item_num = item_num
        self.user_num = user_num
        self.w_lower_bound = w_lower_bound
        self.lambda_ = lambda_

        if isinstance(f, SparseModelMixin):
            self.recom_model = recommender.ClassRecommender(user_num, item_num, self.f)
        elif isinstance(f, SeqModelMixin):
            self.recom_model = recommender.DeepRecommender(user_num, item_num, self.f)

    def obj_func(self, f_prob, w_prob, g_score, f_recom_score, labels):
        if self.w_lower_bound:
            w_prob = torch.clamp(w_prob, min=self.w_lower_bound)
        logloss = -1 * (labels * torch.log(f_prob) +
                        (1 - labels) * torch.log(1 - f_prob)) * w_prob
        diff = w_prob * g_score - f_recom_score * g_score
        obj = logloss + self.lambda_ * diff
        return obj, logloss, diff

    def fit(self,
            tr_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            run_path: Optional[str] = None,
            batch_size: int = 1024, max_len: int = 50,
            min_count: int = 1, max_count: int = 1,
            epoch: int = 10, lr: float = 0.01, decay: float = 0, sample_len: int = 100, cut_len: int = 10,
            cuda: Optional[int] = None):

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        self.f = self.f.to(device)
        self.g = self.g.to(device)
        self.w = self.w.to(device)

        past_hist = tr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()

        if isinstance(self.recom_model, recommender.DeepRecommender):
            hist = tr_df.groupby('uidx').apply(
                lambda x: list(zip(x.ts, x.iidx))).to_dict()
            for k in hist.keys():
                hist[k] = [x[1] for x in sorted(hist[k])]
            self.recom_model.set_user_record(hist)

        item_cnt_dict = tr_df.groupby('iidx').count().uidx.to_dict()
        item_cnt = np.array([item_cnt_dict.get(iidx, 0) for iidx in range(self.item_num)])
        labeled_hist = tr_df.groupby('uidx').apply(
            lambda x: list(zip(x.ts, x.iidx, x.rating))).to_dict()
        for k in labeled_hist.keys():
            labeled_hist[k] = [(x[1], x[2]) for x in sorted(labeled_hist[k])]

        label_dataset = LabeledSequenceData(labeled_hist,
                                            max_len=max_len,
                                            padding_idx=self.item_num,
                                            item_num=self.item_num)
        data_loader = torch.utils.data.DataLoader(
            label_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True)

        writer = SummaryWriter(log_dir=run_path)
        min_optimizer = recommender.build_optimizer(lr, self.f, self.w)
        max_optimizer = recommender.build_optimizer(lr, self.g)

        counter = AlternatingCounter(step_specs={OptimizationStep.ARG_MIN: min_count,
                                                 OptimizationStep.ARG_MAX: max_count})

        def act_func(x):
            return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        iter_ = 0
        for current_epoch in range(epoch):
            run_mode = OptimizationStep.ARG_MIN
            for obs in data_loader:

                user, item_idx, labels, user_hist = list2deivce(obs, device=device)
                user = user.unsqueeze(-1)
                item_idx = item_idx.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
                # item_idx: [B]
                bsz = user.shape[0]

                #  Probability threshold to approximate the quantile function using 100 random negative examples
                negative_samples = torch.randint(0, self.item_num, size=(sample_len,), device=device)
                with torch.no_grad():
                    user_ext = user.repeat(1, sample_len)  # [B, sample_len]
                    negative_samples = negative_samples.repeat(bsz).view(bsz, -1)  # [B, sample_len]
                    recom_score = conditional_forward(user_ext, negative_samples, user_hist, self.f)  # [B, sample_len]
                    sorted_score, _ = torch.sort(recom_score, descending=True)
                    threshold_prob = act_func(sorted_score[:, cut_len])  # [B]

                run_mode = counter.touch()
                if run_mode == OptimizationStep.ARG_MIN:
                    self.g.eval()
                    with torch.no_grad():
                        g_score = conditional_forward(user, item_idx, user_hist, self.g)
                    self.g.train()
                    self.w.train()
                    self.f.train()
                    min_optimizer.zero_grad()
                    w_prob = act_func(conditional_forward(user, item_idx, user_hist, self.w))
                    f_prob = act_func(conditional_forward(user, item_idx, user_hist, self.f))
                    # we need to maximize the objective
                    f_recom = (f_prob > threshold_prob).float()
                    obj, logloss, diff = self.obj_func(f_prob, w_prob, g_score, f_recom, labels)

                    loss = obj.mean()
                    # apply l2 penalty for sparse model only
                    l2 = aggregate_l2(user, item_idx, self.f, self.w, self.g) * decay
                    loss += l2
                    loss.backward()
                    min_optimizer.step()
                elif run_mode == OptimizationStep.ARG_MAX:
                    self.w.eval()
                    self.f.eval()
                    with torch.no_grad():
                        w_prob = act_func(conditional_forward(user, item_idx, user_hist, self.w))
                        f_prob = act_func(conditional_forward(user, item_idx, user_hist, self.f))
                    self.w.train()
                    self.f.train()
                    max_optimizer.zero_grad()
                    g_score = conditional_forward(user, item_idx, user_hist, self.g)
                    f_recom = (f_prob > threshold_prob).float()
                    obj, logloss, diff = self.obj_func(f_prob, w_prob, g_score, f_recom, labels)
                    loss = -1 * obj.mean()
                    l2 = aggregate_l2(user, item_idx, self.f, self.w, self.g) * decay
                    loss += l2
                    loss.backward()
                    max_optimizer.step()
                else:
                    raise NotImplementedError()

                writer.add_scalar('Objective/train', obj.mean().item(), iter_)
                writer.add_scalar('Logloss/train', logloss.mean().item(), iter_)
                writer.add_scalar('Diff/train', diff.mean().item(), iter_)
                writer.add_scalar('L2/train', l2, iter_)
                # Alternate between minimization and maximization.
                iter_ += 1

            if test_df is not None:
                self.f.eval()
                rest = unbiased_eval(self.user_num, self.item_num, test_df, self.recom_model,
                                     rel_model=None,
                                     cut_len=cut_len,
                                     expo_model=None,
                                     past_hist=past_hist)

                writer.add_scalar(f'Recall@{cut_len}/test', rest['recall'], current_epoch)
                writer.add_scalar(f'NDCG@{cut_len}/test', rest['ndcg'], current_epoch)
