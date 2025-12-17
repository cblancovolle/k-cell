import torch
import numpy as np
from torch import Tensor
from cell.common.rls import *
from cell.common.stats import *


class LinearAgent:
    def __init__(
        self,
        ini_X: Tensor,  # (b_size, in_dim)
        ini_y: Tensor,  # (b_size, out_dim)
        forgetting=0.99,
        spatial_factor=1,
    ):
        self.in_dim, self.out_dim = ini_X.size(1), ini_y.size(1)
        ini_X, ini_y = torch.atleast_2d(ini_X), torch.atleast_2d(ini_y)
        self.forgetting = forgetting
        self.spatial_factor = spatial_factor

        self.theta, self.P = rls_init(ini_X, ini_y)
        if ini_X.size(0) > 1:
            self.cov, self.mean = torch.cov(ini_X.T), torch.mean(ini_X, dim=0)
        self.n = ini_X.size(0)

    def spatialization(
        self,
        eps=1e-6,
        dims=None,
    ):
        if dims is not None:
            return (
                self.mean[dims],
                (
                    self.cov.view(self.in_dim, self.in_dim)
                    + eps * torch.eye(self.in_dim)
                )[np.ix_(dims, dims)],
            )
        return (
            self.mean,
            self.cov.view(self.in_dim, self.in_dim) + eps * torch.eye(self.in_dim),
        )

    def predict(self, X_test: Tensor):
        return torch.vmap(rls_predict, in_dims=(None, 0))(
            self.theta, X_test.view(-1, self.in_dim)
        ).view(-1, self.out_dim)

    def predict_one(self, x_test: Tensor):
        return rls_predict(self.theta, x_test.view(self.in_dim)).view(1, self.out_dim)

    def learn_one(
        self,
        x_new: Tensor,
        y_new: Tensor,
        update_model=True,
        update_spatialization=True,
    ):
        if update_model:
            self.theta, self.P = rls_update(
                self.theta,
                self.P,
                x_new.view(self.in_dim),
                y_new,
                forgetting=self.forgetting,
            )
        if update_spatialization:
            self.cov, self.mean = update_covariance_welford(
                self.cov, self.mean, self.n, x_new
            )
            self.n += 1
