import torch
import numpy as np
from torch import Tensor
from copy import deepcopy
from .online_trainer import OnlineTrainer


class IndependentMultiOutputOnlineTrainer:
    def __init__(self, out_dim, trainer_kwargs):
        self.out_dim = out_dim
        self.regressors = [OnlineTrainer(**trainer_kwargs) for i in range(out_dim)]

    @property
    def n_agents(self):
        return np.array([r.n_agents for r in self.regressors])

    def _age(self):
        return [r._age() for r in self.regressors]

    def reset(self):
        [r.reset() for r in self.regressors]

    def learn_one(self, x_new: Tensor, y_new: Tensor):
        return [
            r.learn_one(x_new, y_new[:, idx]) for idx, r in enumerate(self.regressors)
        ]

    def predict_one(self, x_test: Tensor):
        return torch.hstack([r.predict_one(x_test) for r in self.regressors])
