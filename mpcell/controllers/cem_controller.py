import numpy as np
import torch
from numpy import ndarray
from cell.agents.linear_agent import LinearAgent
from cell.trainers.online_trainer import OnlineTrainer
from mpcell.common.constraints import LinearConstraint
from mpcell.wrappers.batch_prediction_wrapper import LinearBatchPredictorWrapper
from torch.distributions import MultivariateNormal
from scipy.stats import chi2


class CEMController:
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
        conservatism_coef=0,
        task_coef=1,
        population_size=64,
        initial_std=1.0,
        k_elites=10,
        jitter=1e-6,
        max_iterations=5,
        predict_deltas=False,
        warmstart_with_previous=True,
        introspection_cost="disagreement",
    ):
        assert model.agent_cls in [LinearAgent]
        assert introspection_cost in ["disagreement", "distance", "activation"]
        self.introspection_cost = introspection_cost
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_dim = state_dim + action_dim
        self.predictor: LinearBatchPredictorWrapper = LinearBatchPredictorWrapper(
            model, predict_deltas=predict_deltas
        )
        self.horizon = horizon
        self.k_elites = k_elites
        self.state_constraints = state_constraints
        self.action_constraints = action_constraints
        self.conservatism_coef = conservatism_coef
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.initial_std = initial_std

        self.cost_fn = cost_fn
        self.cost_kwargs = cost_kwargs
        self.task_coef = task_coef
        self.predict_deltas = predict_deltas
        self.warmstart_with_previous = warmstart_with_previous
        self.jitter = jitter

    def reset(self):
        self.u_prev = None

    def __call__(self, state: ndarray, return_infos=True):
        u_nominal = self.u_prev
        max_states, min_states = self.state_constraints
        max_actions, min_actions = self.action_constraints

        if u_nominal is None:
            u_nominal = torch.zeros((self.horizon, self.action_dim))
        mu = u_nominal.flatten()
        sigma = torch.eye(mu.size(0)) * self.initial_std**2
        current_distrib = MultivariateNormal(mu, covariance_matrix=sigma)

        best_u = None
        best_u_cost = None
        for i in range(self.max_iterations):
            u_samples = current_distrib.sample((self.population_size,)).reshape(
                (self.population_size, self.horizon, self.action_dim)
            )
            u_samples = u_samples.clip(min_actions, max_actions)

            eps = u_samples - u_nominal

            ini_state = np.vstack([state] * self.population_size)
            (
                state_trajectories,
                global_predictions,
                individual_predictions,
                individual_weights,
            ) = self.predictor.predict_trajectory_batch(
                ini_state, u_samples, return_individual_predictions=True
            )  # (pop_size, horizon+1, state_dim), (pop_size, horizon+1, k, state_dim), (pop_size, horizon+1, k, 1)

            if self.introspection_cost == "disagreement":
                gamma = 0.99
                disagreement = (
                    torch.norm(
                        global_predictions.view(
                            self.population_size, self.horizon, 1, self.state_dim
                        )
                        - individual_predictions,
                        p=2,
                        dim=-1,
                    )
                    * individual_weights.view(self.population_size, self.horizon, -1)
                ).sum(
                    dim=2
                )  # (pop_size, horizon) higher = agents do not agree
                discounted_disagreement = (
                    (disagreement * gamma * torch.arange(self.horizon).view(1, -1))
                    .sum(dim=1)
                    .view(self.population_size)
                )  # (pop_size,)
                conservative_costs = discounted_disagreement  # (pop_size,)
            elif self.introspection_cost == "activation":
                _x = torch.cat(
                    (state_trajectories[:, 1:], u_samples), dim=2
                )  # (pop_size, horizon, in_dim)
                (
                    closest_activations,
                    closest_distances,
                ) = self.predictor.batch_closest_activations(
                    _x.reshape(self.population_size * self.horizon, self.in_dim)
                )
                closest_activations = closest_activations.reshape(
                    self.population_size, self.horizon, -1
                )  # (pop_size, horizon, k)
                closest_distances = closest_distances.reshape(
                    self.population_size, self.horizon, -1
                )  # (pop_size, horizon, k)
                max_activations = closest_activations.sum(dim=2)  # (pop_size, horizon)
                conservative_costs = max_activations.sum(dim=1)  # (pop_size,)
            elif self.introspection_cost == "distance":
                _x = torch.cat(
                    (state_trajectories[:, 1:], u_samples), dim=2
                )  # (pop_size, horizon, in_dim)
                (
                    closest_activations,
                    closest_distances,
                ) = self.predictor.batch_closest_activations(
                    _x.reshape(self.population_size * self.horizon, self.in_dim)
                )
                closest_distances = closest_distances.reshape(
                    self.population_size, self.horizon, -1
                )  # (pop_size, horizon, k)
                # distances_score = closest_distances.min(dim=2).values
                distances_score = (
                    closest_distances
                    * individual_weights.view(self.population_size, self.horizon, -1)
                ).sum(dim=-1)
                conservative_costs = distances_score.sum(dim=1)  # (pop_size,)
                # conservative_costs = distances_score.mean(dim=1)  # (pop_size,)
                # conservative_costs = distances_score[:, -1]  # (pop_size,)

            # TASK SPECIFIC COST
            state_trajectories = state_trajectories.clip(min_states, max_states)
            task_costs = self.cost_fn(
                state_trajectories, u_samples, **self.cost_kwargs
            )  # (pop_size,)

            # GLOBAL COST
            costs = (
                self.task_coef * task_costs
                + self.conservatism_coef * conservative_costs
            )  # (pop_size,)

            # Select elites
            elite_costs, elite_idxs = torch.topk(costs, self.k_elites, largest=False)
            mean_elites = torch.mean(u_samples[elite_idxs], dim=0).flatten()
            cov_elites = torch.cov(
                u_samples[elite_idxs].flatten(start_dim=1).T
            ) + self.jitter * torch.eye(self.horizon * self.action_dim)
            current_distrib = MultivariateNormal(
                mean_elites, covariance_matrix=cov_elites
            )

            if (best_u is None) or (best_u_cost > elite_costs[0]):
                best_u = u_samples[elite_idxs[0]]
                best_u_cost = elite_costs[0]

        if self.warmstart_with_previous:
            self.u_prev = torch.vstack([best_u[1:], best_u[-1:]])
        else:
            self.u_prev = None

        if return_infos:
            return best_u.detach().cpu().numpy(), {
                "best_cost": best_u_cost.detach().cpu().numpy(),
            }
        else:
            return best_u.detach().cpu().numpy()
