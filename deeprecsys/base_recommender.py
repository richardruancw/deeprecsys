from typing import List

import pandas as pd
import numpy as np


class Recommender:
    def score(self, u_b: List[int], v_b: List[int]) -> np.ndarray:
        raise NotImplementedError()
        return np.zeros(0)

    def recommend(self, u_s: int, cand_b: List[int], top_k: int) -> List[int]:
        u_b = [u_s] * len(cand_b)
        scores = self.score(u_b, cand_b)
        top_k_ind = scores.argsort()[::-1][:top_k]
        return [cand_b[ind] for ind in top_k_ind]

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()