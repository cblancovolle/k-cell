import torch

from torch import Tensor
from cell.agents.linear_agent import LinearAgent
from cell.common.rls import rls_predict
from cell.common.utils import clipmin_if_all
from ..trainers.online_trainer import OnlineTrainer


class LinearBatchPredictorWrapper:
    def __init__(self, trainer: OnlineTrainer):
        assert trainer.agent_cls in [LinearAgent]
        self.trainer = trainer

    def update_params(self):
        self.params = torch.stack(
            [a.theta for a in self.trainer.agents]
        )  # (n_agents, in_dim + 1, out_dim)

    def predict_batch(self, X_test: Tensor):
        b_size = X_test.size(0)
        self.update_params()
        trainer, params = self.trainer, self.params

        distances = torch.vmap(trainer.distances)(X_test)  # (b_size, n_agents, 1)
        closest_distances, closest_k = torch.topk(
            distances, k=self.trainer.k_closest, dim=1, largest=False
        )  # (b_size, k, 1)
        closest_activations = clipmin_if_all(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        )
        closest_params = params[closest_k.squeeze()]  # (b_size, k, in_dim+1, out_dim)
        if b_size == 1:
            predictions = rls_predict(closest_params, X_test.view(-1)).view(
                1, self.trainer.k_closest, -1
            )  # (b_size, k, out_dim)
        else:
            predictions = torch.vmap(rls_predict)(closest_params, X_test).squeeze(
                2
            )  # (b_size, k, out_dim)
        total_closest_activations = closest_activations.sum(dim=1)  # (b_size, 1)
        weights = closest_activations / total_closest_activations.view(
            -1, 1, 1
        )  # (b_size, k, 1)
        final_prediction = torch.sum(predictions * weights, dim=1)  # (b_size, out_dim)
        return final_prediction
