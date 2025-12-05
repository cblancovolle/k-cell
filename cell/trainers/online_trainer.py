# =============
# Agent Updates
# =============
# if n_neighbors == 0
#   => check if closest has positive loo feedback compared to k-closest
#   => if yes => update model, no shape update
#   => if no => try create agent
# if n_neighbors == 1
#   => check if neighbor has positive loo feedback compared to k-closest
#   => if yes => no model update, update shape
#   => if no => model update, no shape update
#   => update confidence
#   => destroy if needed
# if n_neighbors > 1
#   => check loo feedback for each neighbors
#   => if yes => no model update, update shape
#   => if no => model update, no shape update
#   => update confidences
#   => destroy if needed

import torch
import numpy as np
from scipy.stats import chi2
from torch import Tensor, LongTensor
from collections import deque
from cell.agents import Agent
from cell.common.stats import mahalanobis2


class OnlineTrainer:
    def __init__(
        self,
        in_dim,
        out_dim,
        agent_cls,
        agent_kwargs,
        neighbor_confidence=0.95,
        min_points=3,
        short_term=None,
        confidence_forget=0.1,
        confidence_norm_steepness=3,
        confidence_destroy_th=0.1,
        gain_th=0.0,
        k_closest=4,
        jitter=1e-6,
        kernel_lengthscale=1.0,
    ):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.min_points = min_points
        self.k_closest = k_closest
        self.jitter = jitter
        self.l = kernel_lengthscale
        self.gain_th = gain_th
        self.confidence_forget_lmbd = confidence_forget
        self.confidence_norm_steepness = confidence_norm_steepness
        self.confidence_destroy_threshold = confidence_destroy_th
        self.neighbor_confidence = neighbor_confidence
        self.mahalanobis_neighbor_confidence = torch.as_tensor(
            chi2.ppf(neighbor_confidence, df=self.in_dim)
        )
        self.buffer = deque([], maxlen=min_points)

        self.short_term_buffer = deque(
            [], maxlen=short_term if short_term else min_points * 3
        )
        self.agents: list[Agent] = []

        self.cov = torch.empty((0, in_dim, in_dim))  # for standardization
        self.mean = torch.empty((0, in_dim))  # for standardization
        self.confidence = torch.empty((0, 1))

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def buffer_data(self):
        X, y = map(torch.vstack, zip(*self.buffer))
        return X.double(), y.double()

    @property
    def short_term_buffer_data(self):
        X, y = map(torch.vstack, zip(*self.short_term_buffer))
        return X.double(), y.double()

    def reset(self):
        self.buffer.clear()
        self.short_term_buffer.clear()

    def destroy_agent(self, agent_idxs: LongTensor):
        mask = ~torch.isin(torch.arange(self.n_agents), agent_idxs)
        self.cov = self.cov[mask]
        self.mean = self.mean[mask]
        self.confidence = self.confidence[mask]
        self.agents = [a for id, a in enumerate(self.agents) if id not in agent_idxs]

    def create_agent(self, ini_X: Tensor, ini_y: Tensor):
        new_agent: Agent = self.agent_cls(
            ini_X,
            ini_y,
            **self.agent_kwargs,
        )
        self.confidence = torch.vstack(
            [
                self.confidence,
                torch.zeros((1, 1)),
            ]
        )
        mean, cov = new_agent.spatialization(eps=self.jitter)
        # for standardization
        self.cov = torch.vstack(
            [
                self.cov,
                cov.view(1, self.in_dim, self.in_dim),
            ]
        )
        self.mean = torch.vstack(
            [
                self.mean,
                mean.view(1, self.in_dim),
            ]
        )

        self.agents.append(new_agent)
        self.buffer.clear()

    def _local_baseline(self, x_test):
        baseline_agent: Agent = self.agent_cls(
            *self.short_term_buffer_data,
            **self.agent_kwargs,
        )
        return baseline_agent.predict_one(x_test)

    def _new_agent_baseline(self, X_test):
        new_agent: Agent = self.agent_cls(
            *self.short_term_buffer_data,
            **self.agent_kwargs,
        )
        return new_agent.predict(X_test)

    def distances(self, x_new, agents_idxs=slice(None)):
        means = self.mean[agents_idxs]  # (n_agents, n_features)
        covs = self.cov[agents_idxs]  # (n_agents, n_features, n_features)
        P = torch.linalg.inv(covs)  # (n_agents, n_features, n_features)
        dm2 = torch.vmap(mahalanobis2, in_dims=(None, 0, 0))(
            x_new, P, means
        )  # (n_agents, 1)
        return dm2

    def _predict_one(self, x: Tensor, agents_idxs: LongTensor):
        mus = []
        for agent_id in agents_idxs.view(-1):
            mu = self.agents[agent_id.item()].predict_one(x)
            mus += [mu]
        mus = torch.vstack(mus)
        return mus

    def learn_one(self, x_new: Tensor, y_new: Tensor):
        self.buffer.append((x_new, y_new))
        self.short_term_buffer.append((x_new, y_new))

        if (self.n_agents <= 1) and (len(self.buffer) >= self.min_points):
            self.create_agent(*self.buffer_data)
            return dict(
                n_agents=self.n_agents,
                n_neighbors=0,
                n_created=1,
                n_destroyed=0,
                point_ingested=False,
                min_mahalanobis_dist=np.nan,
                neighbors=[],
            )
        elif self.n_agents <= 1:
            # self.buffer.append((x_new, y_new))
            return dict(
                n_agents=self.n_agents,
                n_neighbors=0,
                n_created=0,
                n_destroyed=0,
                point_ingested=False,
                min_mahalanobis_dist=np.nan,
                neighbors=[],
            )

        # check neighbors
        distances = self.distances(x_new.view(-1)).view(self.n_agents)
        neighbors_mask = (distances <= self.mahalanobis_neighbor_confidence).view(
            self.n_agents
        )
        neighbors = torch.where(neighbors_mask)[0]
        n_neighbors = len(neighbors)
        closest_distances, closest = torch.topk(
            distances, k=min(self.k_closest, self.n_agents), largest=False
        )
        activations = torch.exp(-0.5 * distances / (self.l**2)).view(-1, 1)

        point_has_been_processed = False
        if n_neighbors <= 1:
            agent_to_update = closest[0]

            y_baseline = self._local_baseline(x_new)
            E_base = torch.abs(y_new - y_baseline).mean()
            y_hat = self._predict_one(x_new, agent_to_update)
            E_agent = torch.abs(y_new - y_hat).mean()
            Ci = torch.tanh(
                self.confidence_norm_steepness * (E_base - E_agent) / (E_base + 1e-8)
            )

            if n_neighbors == 1:
                # update confidence
                lmb = self.confidence_forget_lmbd
                self.confidence[agent_to_update] = (
                    self.confidence[agent_to_update] * (1 - lmb) + lmb * Ci.view(-1, 1)
                ).float()

                should_update_model = Ci.item() > 0
                should_update_shape = Ci.item() > 0
            else:
                should_update_model = Ci.item() > 0
                should_update_shape = Ci.item() > 0

            self.agents[agent_to_update].learn_one(
                x_new.view(self.in_dim), y_new, should_update_model, should_update_shape
            )
            self.mean[agent_to_update], self.cov[agent_to_update] = self.agents[
                agent_to_update
            ].spatialization(eps=self.jitter)
            point_has_been_processed = should_update_model or should_update_shape

        if n_neighbors > 1:
            y_propositions = self._predict_one(x_new, neighbors)
            total_activation = torch.sum(activations[neighbors]) + 1e-8
            weights = activations[neighbors] / total_activation
            y_hat = torch.sum(y_propositions * weights, dim=0)
            E = torch.abs(y_new - y_hat).mean()

            Emi = []
            for i in range(n_neighbors):
                mask = torch.arange(n_neighbors) != i
                neighbors_mi = neighbors[mask]
                total_activation_mi = torch.sum(activations[neighbors_mi])
                weights_mi = activations[neighbors_mi] / total_activation_mi
                y_hat_mi = torch.sum(y_propositions[mask] * weights_mi, dim=0)
                Emi += [torch.abs(y_new.view(-1) - y_hat_mi).mean()]

            Emi = torch.stack(Emi)  # (n_neighbors,)
            Ci = torch.tanh(self.confidence_norm_steepness * ((Emi - E) / E))
            # update confidence
            lmb = self.confidence_forget_lmbd
            self.confidence[neighbors] = (
                self.confidence[neighbors] * (1 - lmb) + lmb * Ci.view(-1, 1)
            ).float()

            for idx, agent_to_update in enumerate(neighbors):
                should_update_model = Ci[idx].item() < 0
                should_update_shape = Ci[idx].item() > 0

                self.agents[agent_to_update].learn_one(
                    x_new.view(self.in_dim),
                    y_new,
                    should_update_model,
                    should_update_shape,
                )
                self.mean[agent_to_update], self.cov[agent_to_update] = self.agents[
                    agent_to_update
                ].spatialization(eps=self.jitter)

                point_has_been_processed |= should_update_model or should_update_shape

        n_destroyed = 0
        if n_neighbors >= 1:
            agents_to_destroy = neighbors[
                (
                    self.confidence[neighbors] < -self.confidence_destroy_threshold
                ).squeeze()
            ].view(-1)
        else:
            agents_to_destroy = closest[
                (
                    self.confidence[closest] < -self.confidence_destroy_threshold
                ).squeeze()
            ].view(-1)
        n_destroyed = len(agents_to_destroy)
        self.destroy_agent(agents_to_destroy)

        if (
            (not point_has_been_processed)
            and (n_neighbors == 0)
            and (len(self.buffer) >= self.min_points)
        ):
            X_create, y_create = self.buffer_data
            y_hat = self.predict_many(X_create)
            E_base = torch.abs(y_hat - y_create).mean()

            y_hat_new = self._new_agent_baseline(X_create)
            E_new = torch.abs(y_hat_new - y_create).mean()

            gain = (E_base - E_new) / (E_base + 1e-8)
            if gain > self.gain_th:
                # Compute gain to add an agent
                self.create_agent(*self.buffer_data)

        return dict(
            n_agents=self.n_agents,
            n_neighbors=n_neighbors,
            n_created=int((not point_has_been_processed) and (len(neighbors) == 0)),
            n_destroyed=n_destroyed,
            point_ingested=point_has_been_processed,
            min_mahalanobis_dist=distances.min().numpy(),
            neighbors=neighbors.numpy(),
        )

    def predict_one(self, x_test: Tensor):
        distances = self.distances(x_test.view(-1)).view(-1, 1)
        neighbors_mask = (distances <= self.mahalanobis_neighbor_confidence).view(
            self.n_agents
        )
        neighbors = torch.where(neighbors_mask)[0]
        n_neighbors = len(neighbors)
        _, closest = torch.topk(
            distances.view(-1), k=min(self.k_closest, self.n_agents), largest=False
        )

        agents_to_predict = closest
        y_propositions = self._predict_one(x_test, agents_to_predict)
        activations = (
            torch.exp(-0.5 * distances[agents_to_predict] / (self.l**2)) + 1e-8
        )
        total_activations = torch.sum(activations)
        weights = activations / total_activations

        y_hat = torch.sum(y_propositions * weights, dim=0).view(1, self.out_dim)
        return y_hat

    def predict_many(self, X_test: Tensor):
        y_hat = []
        for x_test in X_test:
            y_hat += [self.predict_one(x_test)]
        return torch.vstack(y_hat)
