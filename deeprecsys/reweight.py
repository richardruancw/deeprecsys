"""Implement the re-weight based learner"""
from typing import Union

import torch
from torch.utils import data
from deeprecsys.module import FactorModel, SeqModelMixin, SparseModelMixin
from deeprecsys.data import NegSequenceData


# from deeprecsys.module import


def conditional_forward(user, item, user_hist, m):
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


class ReWeightLearner:
    def __init__(self, f: torch.nn.Module,
                 w: torch.nn.Module,
                 g: torch.nn.Module, lambda_: float = 1) -> None:
        self.f = f
        self.w = w
        self.g = g
        self.lambda_ = lambda_

    def fit(self, ds: data.Dataset, min_count: int = 1, max_count: int = 1,
            epoch: int = 10, lr: float = 0.01, decay: float = 0):
        """

        :param lr:
        :param ds: training data without negative sampling
        :param min_count:
        :param max_count:
        :param epoch:
        :return:
        """
        min_optimizer = torch.optim.Adam(lr=lr, params=self.f.parameters() + self.w.parameters(()))
        max_optimizer = torch.optim.Adam(lr=lr, params=self.g.parameters())

        for e in range(epoch):
            counter = 0
            run_max = True
            for user, item_idx, labels, user_hist in data:
                if run_max:
                    with torch.no_grad():
                        self.g.eval()
                        g_score = conditional_forward(user, item_idx, user_hist, self.g)
                    self.w.train()
                    self.f.train()
                    max_optimizer.zero_grad()
                    w_prob = torch.logit(conditional_forward(user, item_idx, user_hist, self.w))
                    f_prob = torch.logit(conditional_forward(user, item_idx, user_hist, self.f))
                    # we need to maximize the objective
                    loss = -1 * -1 * (labels * torch.log(f_prob) +
                                 (1 - labels) * torch.log(1 - f_prob)) * w_prob
                    loss += self.lambda_  * (w_prob * g_score - f_prob * g_score)
                    # TODO: Implement the rules of real recommender f

                    loss = -1 * loss.mean()
                    # apply l2 penalty for sparse model only
                    l2 = conditional_l2(user, item_idx, self.w) + conditional_l2(user, item_idx, self.f)
                    loss += l2 * decay
                    loss.backward()
                    max_optimizer.step()
                else:
                    with torch.no_grad():
                        self.w.eval()
                        self.f.eval()
                        w_prob = torch.logit(conditional_forward(user, item_idx, user_hist, self.w))
                        f_prob = torch.logit(conditional_forward(user, item_idx, user_hist, self.f))
                    g_score = conditional_forward(user, item_idx, user_hist, self.g)
                # TODO: Finish the training loop
                # Alternate between minimization and maximization.
                counter += 1
                if run_max and counter >= max_count:
                    run_max = False
                    counter = 0
                elif not run_max and counter >= min_count:
                    run_max = True
                    counter = 1
