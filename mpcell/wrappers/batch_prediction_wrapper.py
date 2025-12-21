from numpy import ndarray
import torch

from torch import Tensor
from cell.agents.linear_agent import LinearAgent
from cell.common.rls import rls_predict

from cell.common.utils import clipmin_if_all, clipmin_first_if_all
from cell.trainers.online_trainer import OnlineTrainer


class LinearBatchPredictorWrapper:
    def __init__(self, trainer: OnlineTrainer, predict_deltas=False):
        assert trainer.agent_cls in [LinearAgent]
        self.trainer = trainer
        self.predict_deltas = predict_deltas

    def update_params(self):
        self.params = torch.stack(
            [a.theta for a in self.trainer.agents]
        )  # (n_agents, in_dim + 1, out_dim)

    def batch_closest_activations(self, X_test: Tensor):
        b_size = X_test.size(0)
        trainer = self.trainer
        distances = torch.vmap(trainer.distances)(X_test)
        closest_distances, closest_k = torch.topk(
            distances,
            k=min(self.trainer.n_agents, self.trainer.k_closest),
            dim=1,
            largest=False,
        )  # (b_size, k, 1)
        closest_activations = torch.vmap(clipmin_if_all)(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        )
        return closest_activations, closest_distances  # (b_size, k)

    def predict_batch(self, X_test: Tensor, return_individual_predictions=False):
        b_size = X_test.size(0)
        trainer, params = self.trainer, self.params

        distances = torch.vmap(trainer.distances)(X_test)  # (b_size, n_agents, 1)
        closest_distances, closest_k = torch.topk(
            distances,
            k=min(self.trainer.n_agents, self.trainer.k_closest),
            dim=1,
            largest=False,
        )  # (b_size, k, 1)
        closest_activations = torch.vmap(clipmin_if_all)(
            torch.exp(-0.5 * closest_distances / (trainer.l**2))
        )
        closest_params = params[closest_k.squeeze()]  # (b_size, k, in_dim+1, out_dim)
        if b_size == 1:
            predictions = rls_predict(closest_params, X_test.view(-1)).view(
                1, min(self.trainer.n_agents, self.trainer.k_closest), -1
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

        if return_individual_predictions:
            return final_prediction, predictions, weights
        return final_prediction

    def predict_trajectory_batch(
        self, state_ini: ndarray, actions: ndarray, return_individual_predictions=False
    ):
        self.update_params()
        assert len(actions.shape) == 3
        assert len(state_ini.shape) == 2
        assert state_ini.shape[0] == actions.shape[0]
        b_size, horizon, action_dim = actions.shape
        _, state_dim = state_ini.shape

        current_state = torch.as_tensor(state_ini, dtype=self.trainer.dtype).view(
            b_size, state_dim
        )
        actions = torch.as_tensor(actions, dtype=self.trainer.dtype).view(
            b_size, horizon, action_dim
        )

        states = [current_state.clone()]
        global_predictions = []
        individual_predictions = []
        weights = []
        for i in range(horizon):
            x_test = torch.hstack(
                (current_state, actions[:, i])
            )  # (b_size, state_dim + action_dim)
            if self.predict_deltas:
                deltas, preds, w = self.predict_batch(
                    x_test, return_individual_predictions=True
                )  # (b_size, state_dim)
                current_state = current_state + deltas
            else:
                current_state, preds, w = self.predict_batch(
                    x_test, individual_predictions=True
                )  # (b_size, state_dim)
            states += [current_state.clone()]
            global_predictions += [deltas]
            individual_predictions += [preds]
            weights += [w]

        if return_individual_predictions:
            return (
                torch.stack(states, dim=1),
                torch.stack(global_predictions, dim=1),  # (b_size, horizon, state_dim)
                torch.stack(
                    individual_predictions, dim=1
                ),  # (b_size, horizon, k, state_dim)
                torch.stack(weights, dim=1),  # (b_size, horizon, k, 1)
            )
        return torch.stack(states, dim=1)
