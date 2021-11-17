"""Implement the re-weight based learner"""
import contextlib
from bisect import bisect
from enum import Enum
from typing import Optional, Dict, Tuple, List, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from deeprecsys import recommender
from deeprecsys.data import LabeledSequenceData
from deeprecsys.eval import unbiased_eval, unbiased_full_eval
from deeprecsys.module import SeqModelMixin, SparseModelMixin


class OptimizationStep(Enum):
    ARG_MIN = 0
    ARG_MAX = 1
    ARG_MIN_W = 2
    ARG_MIN_F = 3
    ARG_MAX_G = 4


def conditional_forward(user, item, user_hist, m: torch.nn.Module, inference: bool = False):
    """Sparse model use (user, item), seq model use (item, user_hist)
    """
    ctx_manager = contextlib.nullcontext()
    if inference:
        m.eval()
        ctx_manager = torch.no_grad()
    with ctx_manager:
        if isinstance(m, SeqModelMixin):
            out = m(item, user_hist)
        elif isinstance(m, SparseModelMixin):
            out = m(user, item)
        else:
            raise ValueError('Model type not supported')
    if inference:
        m.train()
    return out


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


class LoopCounter:
    def __init__(self, loop_specs: List[Tuple[Enum, int]]):
        self._slots = []
        self._fill_slots(self._slots, [x for x in loop_specs if x[1] > 0])

    def _fill_slots(self, out: List[Enum], candidates: List[Tuple[Enum, int]]):
        if not candidates:
            return

        step = candidates[0][0]
        repeat = candidates[0][1]
        for _ in range(repeat):
            out.append(step)
            self._fill_slots(out, candidates[1:])

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx < len(self._slots):
            idx = self.curr_idx
            self.curr_idx += 1
            return self._slots[idx]
        else:
            raise StopIteration


class LoopManager:
    def __init__(self, cache):
        self.cache = cache
        self.data_iter = {}
        for k in self.cache.keys():
            self.data_iter[k] = iter(self.cache[k]['data'])

    def get_data(self, step: Enum):
        try:
            x = next(self.data_iter[step])
        except StopIteration:
            self.data_iter[step] = iter(self.cache[step]['data'])
            x = next(self.data_iter[step])
        return x

    def get_optimizer(self, step: Enum):
        return self.cache[step]['optimizer']


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

        if isinstance(f, SparseModelMixin):
            self.rel_model = recommender.ClassRecommender(user_num, item_num, self.w)
        elif isinstance(f, SeqModelMixin):
            self.rel_model = recommender.DeepRecommender(user_num, item_num, self.w)

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
            epoch: int = 10, max_lr: float = 0.01, min_lr: float = 0.01, decay: float = 0, sample_len: int = 100,
            cut_len: int = 10,
            cuda: Optional[int] = None, topk: int = 100, true_rel_model: Optional[recommender.Recommender] = None):

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        self.f = self.f.to(device)
        self.g = self.g.to(device)
        self.w = self.w.to(device)

        max_len = max_len if isinstance(self.f, SeqModelMixin) else 1
        allow_empty = isinstance(self.f, SparseModelMixin)

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
                                            window=True,
                                            padding_idx=self.item_num,
                                            past_hist=past_hist,
                                            item_num=self.item_num,
                                            allow_empty=allow_empty)
        data_loader = torch.utils.data.DataLoader(
            label_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)

        writer = SummaryWriter(log_dir=run_path)
        min_optimizer = recommender.build_optimizer(min_lr, self.f, self.w)
        max_optimizer = recommender.build_optimizer(max_lr, self.g)

        counter = AlternatingCounter(step_specs={OptimizationStep.ARG_MIN: min_count,
                                                 OptimizationStep.ARG_MAX: max_count})

        def act_func(x):
            return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        iter_ = 0
        for current_epoch in range(epoch):
            for obs in data_loader:
                user, item_idx, labels, user_hist = list2deivce(obs, device=device)
                user = user.unsqueeze(-1)
                item_idx = item_idx.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
                # item_idx: [B]
                bsz = user.shape[0]

                #  Probability threshold to approximate the quantile function using 100 random negative examples
                negative_samples = torch.randint(0, self.item_num, size=(sample_len,), device=device)
                self.f.eval()
                with torch.no_grad():
                    user_ext = user.repeat(1, sample_len)  # [B, sample_len]
                    negative_samples = negative_samples.repeat(bsz).view(bsz, -1)  # [B, sample_len]
                    recom_score = conditional_forward(user_ext, negative_samples, user_hist, self.f)  # [B, sample_len]
                    sorted_score, _ = torch.sort(recom_score, descending=True)
                    threshold_prob = act_func(sorted_score[:, cut_len])  # [B]
                self.f.train()

                run_mode = counter.touch()
                if run_mode == OptimizationStep.ARG_MIN:
                    self.g.eval()
                    with torch.no_grad():
                        g_score = conditional_forward(user, item_idx, user_hist, self.g)
                    self.g.train()
                    min_optimizer.zero_grad()
                    self.w.train()
                    self.f.train()
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
                # rest = unbiased_eval(self.user_num, self.item_num, test_df, self.recom_model,
                #                      rel_model=None,
                #                      cut_len=cut_len,
                #                      expo_model=None,
                #                      past_hist=past_hist)
                #
                # writer.add_scalar(f'Recall@{cut_len}/test', rest['recall'], current_epoch)
                # writer.add_scalar(f'NDCG@{cut_len}/test', rest['ndcg'], current_epoch)

                rel_score = unbiased_full_eval(self.user_num, self.item_num, self.recom_model, topk=topk,
                                               dat_df=test_df,
                                               rel_model=true_rel_model)
                writer.add_scalar(f'full_top_{topk}_relevance', rel_score, current_epoch)


class ReWeightLearnerV2:
    def __init__(self, f: torch.nn.Module,
                 w: torch.nn.Module,
                 g: torch.nn.Module,
                 item_num: int, user_num: int,
                 lambda_: float = 1,
                 w_lower_bound: Optional[float] = None,
                 ) -> None:
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

        if isinstance(w, SparseModelMixin):
            self.rel_model = recommender.ClassRecommender(user_num, item_num, self.w)
        elif isinstance(w, SeqModelMixin):
            self.rel_model = recommender.DeepRecommender(user_num, item_num, self.w)

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
            batch_size: int = 1024,
            max_len: int = 50,
            f_count: int = 1, w_count: int = 1, g_count: int = 1,
            epoch: int = 10,
            f_lr: float = 0.01, w_lr: float = 0.01, g_lr: float = 0.01,
            decay: float = 0, sample_len: int = 100, cut_len: int = 10,
            cuda: Optional[int] = None,
            topk: int = 100,
            true_rel_model: Optional[recommender.Recommender] = None,
            past_hist: Optional[Dict[int, Set[int]]] = None):

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        self.f = self.f.to(device)
        self.g = self.g.to(device)
        self.w = self.w.to(device)

        max_len = max_len if isinstance(self.f, SeqModelMixin) else 1
        allow_empty = isinstance(self.f, SparseModelMixin)

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

        def get_data_loader():
            label_dataset = LabeledSequenceData(labeled_hist,
                                                max_len=max_len,
                                                window=True,
                                                padding_idx=self.item_num,
                                                past_hist=past_hist,
                                                item_num=self.item_num,
                                                allow_empty=allow_empty)
            data_loader = torch.utils.data.DataLoader(
                label_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True)
            return data_loader

        writer = SummaryWriter(log_dir=run_path)
        f_optimizer = recommender.build_optimizer(f_lr, self.f)
        w_optimizer = recommender.build_optimizer(w_lr, self.w)
        g_optimizer = recommender.build_optimizer(g_lr, self.g)

        f_data = get_data_loader()
        g_data = get_data_loader()
        w_data = get_data_loader()

        cache = {
            OptimizationStep.ARG_MIN_F: {'data': f_data, 'optimizer': f_optimizer},
            OptimizationStep.ARG_MAX_G: {'data': g_data, 'optimizer': g_optimizer},
            OptimizationStep.ARG_MIN_W: {'data': w_data, 'optimizer': w_optimizer}
        }

        counter = LoopCounter([(OptimizationStep.ARG_MIN_F, f_count),
                               (OptimizationStep.ARG_MIN_W, w_count),
                               (OptimizationStep.ARG_MAX_G, g_count), ])

        loop_manager = LoopManager(cache)

        def act_func(x):
            return torch.sigmoid(torch.clamp(x, min=-8, max=8))

        iter_ = 0
        for current_epoch in range(epoch):
            for _ in range(len(f_data)):
                for step in counter:
                    obs = loop_manager.get_data(step)
                    optimizer = loop_manager.get_optimizer(step)

                    user, item_idx, labels, user_hist = list2deivce(obs, device=device)

                    user = user.unsqueeze(-1)
                    item_idx = item_idx.unsqueeze(-1)
                    labels = labels.unsqueeze(-1)
                    # item_idx: [B]
                    bsz = user.shape[0]
                    #  Probability threshold to approximate the quantile function using 100 random negative examples
                    negative_samples = torch.randint(0, self.item_num, size=(sample_len,), device=device)
                    user_ext = user.repeat(1, sample_len)  # [B, sample_len]
                    negative_samples = negative_samples.repeat(bsz).view(bsz, -1)  # [B, sample_len]
                    recom_score = conditional_forward(user_ext, negative_samples, user_hist, self.f,
                                                      inference=True)  # [B, sample_len]
                    sorted_score, _ = torch.sort(recom_score, descending=True)
                    threshold_prob = act_func(sorted_score[:, cut_len])  # [B]

                    optimizer.zero_grad()

                    # use inference (no_grad + eval) mode if current step does not optimize it.
                    g_score = conditional_forward(user, item_idx, user_hist, self.g,
                                                  inference=step != OptimizationStep.ARG_MAX_G)
                    w_prob = act_func(conditional_forward(user, item_idx, user_hist, self.w,
                                                          inference=step != OptimizationStep.ARG_MIN_W))
                    f_prob = act_func(conditional_forward(user, item_idx, user_hist, self.f,
                                                          inference=step != OptimizationStep.ARG_MIN_F))

                    f_recom = (f_prob > threshold_prob).float()
                    obj, logloss, diff = self.obj_func(f_prob, w_prob, g_score, f_recom, labels)

                    # in maximization step need to flip the sign
                    sign = -1.0 if step == OptimizationStep.ARG_MAX_G else 1.0
                    loss = obj.mean() * sign

                    # apply l2 penalty for sparse model only
                    l2 = aggregate_l2(user, item_idx, self.f, self.w, self.g) * decay
                    loss += l2
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('Objective/train', obj.mean().item(), iter_)
                    writer.add_scalar('Logloss/train', logloss.mean().item(), iter_)
                    writer.add_scalar('Diff/train', diff.mean().item(), iter_)
                    writer.add_scalar('L2/train', l2, iter_)
                    # Alternate between minimization and maximization.
                    iter_ += 1

            if test_df is not None:
                self.f.eval()
                # rest = unbiased_eval(self.user_num, self.item_num, test_df, self.recom_model,
                #                      rel_model=None,
                #                      cut_len=cut_len,
                #                      expo_model=None,
                #                      past_hist=past_hist)
                #
                # writer.add_scalar(f'Recall@{cut_len}/test', rest['recall'], current_epoch)
                # writer.add_scalar(f'NDCG@{cut_len}/test', rest['ndcg'], current_epoch)

                rel_score = unbiased_full_eval(self.user_num, self.item_num, self.recom_model, topk=topk,
                                               dat_df=test_df,
                                               rel_model=true_rel_model,
                                               past_hist=past_hist)
                writer.add_scalar(f'full_top_{topk}_relevance', rel_score, current_epoch)


class ReWeightLearnerV3:
    def __init__(self, f: torch.nn.Module,
                 w: torch.nn.Module,
                 g: torch.nn.Module,
                 item_num: int, user_num: int,
                 lambda_: float = 1,
                 w_lower_bound: Optional[float] = None,
                 ) -> None:
        self.f = f
        self.w = w
        self.g = g
        self.item_num = item_num
        self.user_num = user_num
        self.w_lower_bound = w_lower_bound
        self.lambda_ = lambda_
        self.sigmoid = torch.nn.Sigmoid()

        if isinstance(f, SparseModelMixin):
            self.recom_model = recommender.ClassRecommender(user_num, item_num, self.f)
        elif isinstance(f, SeqModelMixin):
            self.recom_model = recommender.DeepRecommender(user_num, item_num, self.f)

        if isinstance(w, SparseModelMixin):
            self.rel_model = recommender.ClassRecommender(user_num, item_num, self.w)
        elif isinstance(w, SeqModelMixin):
            self.rel_model = recommender.DeepRecommender(user_num, item_num, self.w)

    def act_func(self, x):
        return torch.sigmoid(torch.clamp(x, min=-8, max=8))

    def obj_func(self, f_prob, w_prob, g_score, f_recom_score, labels, mask=0):
        # use mask to determine if the logloss will be counted when updating w

        assert mask in [0, 1]
        logloss = self.logloss(f_prob, w_prob, labels)
        diff = g_score * f_recom_score * torch.log(w_prob)
        obj = mask * logloss + self.lambda_ * diff
        return obj, logloss, diff

    def logloss(self, f_prob, w_prob, labels):
        logloss = -1 * (labels * torch.log(f_prob) +
                        (1 - labels) * torch.log(1 - f_prob)) * w_prob
        return logloss

    def fit(self,
            tr_df: pd.DataFrame,
            val_df: Optional[pd.DataFrame] = None,
            run_path: Optional[str] = None,
            batch_size: int = 1024,
            max_len: int = 50,
            f_count: int = 1, w_count: int = 1, g_count: int = 1,
            epoch: int = 10,
            f_lr: float = 0.01, w_lr: float = 0.01, g_lr: float = 0.01,
            decay: float = 0, sample_len: int = 100, cut_len: int = 10,
            fast_train: Optional[bool] = False,
            cuda: Optional[int] = None,
            topk: int = 100,
            true_rel_model: Optional[recommender.Recommender] = None,
            past_hist: Optional[Dict[int, Set[int]]] = None,
            labeled_hist=None):

        if run_path:
            writer = SummaryWriter(log_dir=run_path)
        else:
            writer = None

        if cuda is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{cuda}')

        self.f = self.f.to(device)
        self.g = self.g.to(device)
        self.w = self.w.to(device)

        max_len = max_len if isinstance(self.f, SeqModelMixin) else 1
        allow_empty = isinstance(self.f, SparseModelMixin)

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

        def get_data_loader():
            label_dataset = LabeledSequenceData(labeled_hist,
                                                max_len=max_len,
                                                window=True,
                                                padding_idx=self.item_num,
                                                past_hist=past_hist,
                                                item_num=self.item_num,
                                                allow_empty=allow_empty)
            data_loader = torch.utils.data.DataLoader(
                label_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True)
            return data_loader

        f_optimizer = recommender.build_optimizer(f_lr, self.f)
        w_optimizer = recommender.build_optimizer(w_lr, self.w)
        g_optimizer = recommender.build_optimizer(g_lr, self.g)

        f_data = get_data_loader()
        g_data = get_data_loader()
        w_data = get_data_loader()

        cache = {
            OptimizationStep.ARG_MIN: {'data': f_data},
            OptimizationStep.ARG_MAX: {'data': w_data}
        }

        counter = LoopCounter([(OptimizationStep.ARG_MIN, f_count),
                               (OptimizationStep.ARG_MAX, w_count), ])

        loop_manager = LoopManager(cache)

        iter_ = 0
        print('start training')

        for current_epoch in range(epoch):
            for batch_id in range(len(f_data)):
                # obs = loop_manager.get_data(OptimizationStep.ARG_MIN_F)
                for step in counter:
                    obs = loop_manager.get_data(step)

                    user, item_idx, labels, user_hist = list2deivce(obs, device=device)

                    user = user.unsqueeze(-1)
                    item_idx = item_idx.unsqueeze(-1)
                    labels = labels.unsqueeze(-1)
                    # item_idx: [B]
                    bsz = user.shape[0]
                    #  Probability threshold to approximate the quantile function using 100 random negative examples
                    negative_samples = torch.randint(0, self.item_num, size=(sample_len,), device=device)
                    user_ext = user.repeat(1, sample_len)  # [B, sample_len]
                    negative_samples = negative_samples.repeat(bsz).view(bsz, -1)  # [B, sample_len]
                    recom_score = conditional_forward(user_ext, negative_samples, user_hist, self.f,
                                                      inference=True)  # [B, sample_len]
                    sorted_score, _ = torch.sort(recom_score, descending=True)
                    threshold_prob = self.act_func(sorted_score[:, cut_len])  # [B]

                    # use inference (no_grad + eval) mode if current step does not optimize it.
                    if step == OptimizationStep.ARG_MAX:  # update w
                        w_optimizer.zero_grad()
                        w_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.w,
                                                                   inference=False))
                        if self.w_lower_bound:
                            w_prob = torch.clamp(w_prob, min=self.w_lower_bound)

                        if fast_train:  # set g_score to -1 for fast training
                            g_score = torch.ones_like(w_prob) * -1
                        else:
                            g_score = torch.tanh(
                                conditional_forward(user, item_idx, user_hist, self.g, inference=True))

                        f_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.f,
                                                                   inference=True))
                        f_recom = (f_prob > threshold_prob).float()
                        obj, logloss, diff = self.obj_func(f_prob, w_prob, g_score, f_recom, labels)
                        loss = obj.mean()

                        # apply l2 penalty for sparse model only
                        l2 = aggregate_l2(user, item_idx, self.w) * decay
                        loss += l2
                        loss.backward()
                        w_optimizer.step()

                    if step == OptimizationStep.ARG_MAX and not fast_train:  # update g
                        g_optimizer.zero_grad()
                        g_score_raw = conditional_forward(user, item_idx, user_hist, self.g, inference=False)
                        g_score = torch.tanh(g_score_raw)

                        w_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.w,
                                                                   inference=True))
                        if self.w_lower_bound:
                            w_prob = torch.clamp(w_prob, min=self.w_lower_bound)

                        f_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.f,
                                                                   inference=True))
                        f_recom = (f_prob > threshold_prob).float()
                        obj, logloss, diff = self.obj_func(f_prob, w_prob, g_score, f_recom, labels)
                        loss = -1 * obj.mean()

                        # apply l2 penalty for sparse model only
                        l2 = aggregate_l2(user, item_idx, self.g) * decay
                        loss += l2
                        loss.backward()
                        g_optimizer.step()

                    if step == OptimizationStep.ARG_MIN:  # update f
                        f_optimizer.zero_grad()
                        w_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.w,
                                                                   inference=True))
                        if self.w_lower_bound:
                            w_prob = torch.clamp(w_prob, min=self.w_lower_bound)

                        f_prob = self.act_func(conditional_forward(user, item_idx, user_hist, self.f,
                                                                   inference=False))
                        logloss = self.logloss(f_prob, w_prob, labels)
                        loss = logloss.mean()

                        l2 = aggregate_l2(user, item_idx, self.f) * decay
                        loss += l2
                        loss.backward()
                        f_optimizer.step()

                    if writer:
                        writer.add_scalar('Objective/train', obj.mean().item(), iter_)
                        writer.add_scalar('Logloss/train', logloss.mean().item(), iter_)
                        writer.add_scalar('Diff/train', diff.mean().item(), iter_)
                        writer.add_scalar('L2/train', l2, iter_)

                    # Alternate between minimization and maximization.
                    iter_ += 1

            if val_df is not None:
                self.f.eval()
                rest = unbiased_eval(self.user_num, self.item_num, val_df, self.recom_model,
                                     rel_model=None,
                                     expo_model=None,
                                     past_hist=past_hist)
                #
                # writer.add_scalar(f'Recall@{cut_len}/test', rest['recall'], current_epoch)
                # writer.add_scalar(f'NDCG@{cut_len}/test', rest['ndcg'], current_epoch)

                rel_score = unbiased_full_eval(self.user_num, self.item_num, self.recom_model, topk=topk,
                                               dat_df=val_df,
                                               rel_model=true_rel_model,
                                               past_hist=None)
                # writer.add_scalar(f'full_top_{topk}_relevance', rel_score, current_epoch)
                # val_rec.append([rest['recall'], rest['ndcg'], rel_score])
                print('epoch:{}, recall:{}, ndcg:{}, rel:{}'.format(current_epoch, rest['recall'], rest['ndcg'],
                                                                    rel_score))