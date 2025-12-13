from numpy import ndarray
import torch

from torch import Tensor
from cell.agents.linear_agent import LinearAgent

from cell.common.utils import clipmin_if_all
from cell.trainers.online_trainer import OnlineTrainer


class StateActionLinearizerWrapper:
    def __init__(
        self,
        trainer: OnlineTrainer,
        state_dim: int,
        action_dim: int,
    ):
        assert trainer.agent_cls in [LinearAgent]
        assert trainer.in_dim == state_dim + action_dim
        assert trainer.out_dim == state_dim
        self.trainer = trainer
        self.state_dim = state_dim
        self.action_dim = action_dim

    def update_params(self):
        self.params = torch.stack(
            [a.theta for a in self.trainer.agents]
        )  # (n_agents, in_dim + 1, out_dim)

    def local_model_one(
        self,
        X_test: Tensor,  # (in_dim,)
    ):
        X_test = torch.as_tensor(X_test, dtype=self.trainer.dtype)
        assert len(X_test.size()) == 1
        self.update_params()
        trainer, params = self.trainer, self.params

        distances = trainer.distances(X_test).T  # (n_agents, 1)
        closest_distances, closest_k = torch.topk(
            distances, k=self.trainer.k_closest, largest=False
        )  # (k, 1)
        closest_activations = clipmin_if_all(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        ).T
        closest_params = params[closest_k.squeeze()]  # (k, in_dim+1, out_dim)
        total_closest_activations = closest_activations.sum()  # (1,)
        weights = closest_activations / total_closest_activations  # (k, 1)
        local_params = (weights.view(-1, 1, 1) * closest_params).sum(dim=0)

        closest_means = self.trainer.mean[closest_k].view(
            -1, self.trainer.in_dim
        )  # (k, in_dim)
        closest_covariances = self.trainer.cov[closest_k].view(
            -1, self.trainer.in_dim, self.trainer.in_dim
        )  # (k, in_dim, in_dim)
        # we use Toeplitz decomposition to ensure covariance is symmetric
        # (https://en.wikipedia.org/wiki/Toeplitz_matrix)
        closest_covariances = 0.5 * (
            closest_covariances + closest_covariances.transpose(-1, -2)
        )

        return (
            local_params[: self.state_dim].numpy(),  # (state_dim, state_dim)
            local_params[self.state_dim : -1].numpy(),  # (action_dim, state_dim)
            local_params[-1:].numpy(),  # (1, state_dim)
        ), (closest_means.numpy(), closest_covariances.numpy())

    def local_model_many(self, X_test: Tensor | ndarray):
        X_test = torch.as_tensor(X_test, dtype=self.trainer.dtype)
        assert len(X_test.size()) == 2
        self.update_params()
        b_size = X_test.size(0)
        trainer, params = self.trainer, self.params

        distances = torch.vmap(trainer.distances)(X_test).view(b_size, trainer.n_agents)
        closest_distances, closest_k = torch.topk(
            distances, k=self.trainer.k_closest, largest=False, dim=1
        )  # (b_size, k)
        closest_activations = torch.vmap(clipmin_if_all)(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        )  # (b_size, k)
        closest_params = params[closest_k]  # (b_size, k, in_dim + 1, out_dim)
        total_closest_activations = closest_activations.sum(
            dim=1, keepdims=True
        )  # (b_size, 1)
        weights = closest_activations / total_closest_activations  # (b_size, k)
        local_params = (weights.view(b_size, -1, 1, 1) * closest_params).sum(
            dim=1
        )  # (b_size, in_dim+1, out_dim)

        closest_means = self.trainer.mean[closest_k]  # (b_size, k, in_dim)
        closest_covariances = self.trainer.cov[closest_k]  # (b_size, k, in_dim, in_dim)
        # we use Toeplitz decomposition to ensure covariance is symmetric
        # (https://en.wikipedia.org/wiki/Toeplitz_matrix)
        closest_covariances = 0.5 * (
            closest_covariances + closest_covariances.transpose(-1, -2)
        )

        return (
            local_params[:, : self.state_dim].numpy(),  # (b_size, state_dim, state_dim)
            local_params[
                :, self.state_dim : -1
            ].numpy(),  # (b_size, action_dim, state_dim)
            local_params[:, -1:].numpy(),  # (b_size, 1, state_dim)
        ), (closest_means.numpy(), closest_covariances.numpy())
