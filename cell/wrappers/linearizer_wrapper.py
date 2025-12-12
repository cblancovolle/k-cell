import torch

from torch import Tensor
from cell.agents.linear_agent import LinearAgent
from cell.common.utils import clipmin_if_all
from ..trainers.online_trainer import OnlineTrainer


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

    def matrices(
        self,
        X_test: Tensor,  # (in_dim,)
    ):
        assert len(X_test.size()) == 1
        b_size = X_test.size(0)
        self.update_params()
        trainer, params = self.trainer, self.params

        distances = trainer.distances(X_test).T  # (n_agents, 1)
        print(distances.shape)
        closest_distances, closest_k = torch.topk(
            distances, k=self.trainer.k_closest, largest=False
        )  # (b_size, k, 1)
        closest_activations = clipmin_if_all(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        ).T
        closest_params = params[closest_k.squeeze()]  # (b_size, k, in_dim+1, out_dim)
        total_closest_activations = closest_activations.sum()  # (b_size, 1)
        weights = closest_activations / total_closest_activations  # (b_size, k, 1)
        print(
            closest_activations.shape,
            weights.shape,
            closest_params.shape,
        )
        local_params = (weights.view(-1, 1, 1) * closest_params).sum(dim=0)
        return (
            local_params[: self.state_dim],  # (state_dim, state_dim)
            local_params[self.state_dim : -1],  # (action_dim, state_dim)
            local_params[-1:],  # (1, state_dim)
        )
