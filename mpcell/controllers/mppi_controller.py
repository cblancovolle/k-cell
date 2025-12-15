import numpy as np
import torch
from numpy import ndarray
from cell.agents.linear_agent import LinearAgent
from cell.trainers.online_trainer import OnlineTrainer
from mpcell.common.constraints import LinearConstraint
from mpcell.wrappers.batch_prediction_wrapper import LinearBatchPredictorWrapper
from torch.distributions import MultivariateNormal


class MPPIController:
    def __init__(
        self,
        model: OnlineTrainer,
        state_dim,
        action_dim,
        horizon,
        cost_fn,
        cost_kwargs={},
        state_constraints=(1, -1),
        action_constraints=(1, -1),
        conservatism_coef=1e-3,
        population_size=64,
        initial_std=1.0,
        temperature=0.3,
    ):
        assert model.agent_cls in [LinearAgent]
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_dim = state_dim + action_dim
        self.predictor: LinearBatchPredictorWrapper = LinearBatchPredictorWrapper(model)
        self.horizon = horizon
        self.state_constraints = state_constraints
        self.action_constraints = action_constraints
        self.conservatism_coef = conservatism_coef
        self.population_size = population_size
        self.initial_std = initial_std

        self.cost_fn = cost_fn
        self.cost_kwargs = cost_kwargs
        self.temperature = temperature

    def reset(self):
        self.u_prev = None

    def __call__(self, state: ndarray, return_infos=True):
        u_nominal = self.u_prev
        max_states, min_states = self.state_constraints
        max_actions, min_actions = self.action_constraints

        if u_nominal is None:
            u_nominal = torch.zeros((self.horizon, self.action_dim))
        loc = u_nominal.flatten()
        current_distrib = MultivariateNormal(
            loc, covariance_matrix=torch.eye(loc.size(0)) * self.initial_std**2
        )

        u_samples = current_distrib.sample((self.population_size,)).reshape(
            (self.population_size, self.horizon, self.action_dim)
        )
        u_samples = u_samples.clip(min_actions, max_actions)

        eps = u_samples - u_nominal

        ini_state = np.vstack([state] * self.population_size)

        state_trajectories = self.predictor.predict_trajectory_batch(
            ini_state, u_samples
        )  # (pop_size, horizon+1, state_dim)

        # CONSERVATIVE COST
        _x = torch.cat(
            (state_trajectories[:, 1:], u_samples), dim=2
        )  # (pop_size, horizon, in_dim)
        closest_activations = self.predictor.batch_closest_activations(
            _x.reshape(self.population_size * self.horizon, self.in_dim)
        ).reshape(
            self.population_size, self.horizon, -1
        )  # (pop_size, horizon, k)
        max_activations = closest_activations.max(dim=-1).values  # (pop_size, horizon)
        conservative_costs = max_activations.mean(dim=-1)  # (pop_size,)

        # TASK SPECIFIC COST
        state_trajectories = state_trajectories.clip(min_states, max_states)
        task_costs = self.cost_fn(
            state_trajectories, u_samples, **self.cost_kwargs
        )  # (pop_size,)

        # GLOBAL COST
        costs = task_costs + self.conservatism_coef * conservative_costs
        min_cost = torch.min(costs)

        _lambda = self.temperature
        weights = torch.exp(-(costs - min_cost) / _lambda)
        weights = weights / torch.sum(weights)
        delta_u = torch.sum(weights[:, None, None] * eps, dim=0)
        u_updated = u_nominal + delta_u

        self.u_prev = np.vstack([u_updated[1:], u_updated[-1:]])

        if return_infos:
            return u_updated.detach().cpu().numpy(), {
                "costs": costs.detach().cpu().numpy(),
                "conservative_cost": conservative_costs.detach().cpu().numpy(),
                "task_cost": task_costs.detach().cpu().numpy(),
                "weights": weights.detach().cpu().numpy(),
            }
        else:
            return u_updated.detach().cpu().numpy()
