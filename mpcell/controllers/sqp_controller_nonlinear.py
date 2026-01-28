from numpy import ndarray
from cell.agents.linear_agent import LinearAgent
from cell.trainers.online_trainer import OnlineTrainer
from mpcell.common.constraints import LinearConstraint
from mpcell.common.utils import quad_form
from mpcell.wrappers.linearizer_wrapper import StateActionLinearizerWrapper
from casadi import *

from mpcell.wrappers.state_action_prediction_wrapper import StateActionPredictorWrapper


class SQPController:
    def __init__(
        self,
        model: OnlineTrainer,
        state_dim,
        action_dim,
        horizon,
        state_constraints=[],
        action_constraints=[],
        conservatism_coef=1e-3,
        task_coef=1.0,
        solver="osqp",
        error_on_fail=False,
        max_sqp_iters=5,
        alpha_decay=0.5,
        min_alpha=1e-3,
        lin_error_tol=0.1,
        predict_deltas=False,
        line_search=True,
    ):
        assert model.agent_cls in [LinearAgent]
        assert solver in ["osqp", "ipopt"]
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.in_dim = state_dim + action_dim
        self.predict_deltas = predict_deltas
        self.linearizer: StateActionLinearizerWrapper = StateActionLinearizerWrapper(
            model, state_dim, action_dim
        )
        self.predictor: StateActionPredictorWrapper = StateActionPredictorWrapper(
            model, predict_deltas=self.predict_deltas
        )
        self.horizon = horizon

        self.state_constraints: list[LinearConstraint] = state_constraints
        self.action_constraints: list[LinearConstraint] = action_constraints
        self.conservatism_coef = conservatism_coef
        self.task_coef = task_coef
        self.solver = solver
        self.error_on_fail = error_on_fail
        self.max_sqp_iter = max_sqp_iters
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.lin_error_tol = lin_error_tol
        self.line_search = line_search

    def reset(self):
        self.u_prev = None

    def build_problem(self, cost_fn, cost_kwargs):
        horizon = self.horizon
        state_dim = self.state_dim
        action_dim = self.action_dim
        in_dim = self.in_dim

        if self.solver == "osqp":
            opti = Opti("conic")
            opti.solver(
                "osqp",
                {
                    "error_on_fail": self.error_on_fail,
                    # "warm_start_dual": False,
                    # "warm_start_primal": False,
                },
            )
        else:
            opti = Opti()
            opti.solver(
                "ipopt",
                {
                    "error_on_fail": self.error_on_fail,
                    "ipopt.print_level": 0,
                    "ipopt.sb": "yes",
                    "ipopt.max_iter": 100,
                    "print_time": 0,
                },
            )

        # Symbolic Parameters
        x_prev = opti.parameter(horizon + 1, state_dim)
        x0 = opti.parameter(state_dim)

        A = opti.parameter(
            horizon, state_dim * state_dim
        )  # -> (horizon+1, state_dim, state_dim)
        B = opti.parameter(horizon, action_dim * state_dim)
        c = opti.parameter(horizon, 1 * state_dim)

        means = opti.parameter(horizon, self.model.k_closest * self.model.n_spatial)
        precisions = opti.parameter(
            horizon, self.model.k_closest * self.model.n_spatial * self.model.n_spatial
        )

        # Symbolic Variables
        x = opti.variable(horizon + 1, state_dim)
        u = opti.variable(horizon, action_dim)

        # Initial state constraint
        opti.subject_to(x[0, :].T == x0)

        # Dynamics Constraints
        for t in range(horizon):
            xt = x[t, :].T
            ut = u[t, :].T
            At = A[t, :].reshape((state_dim, state_dim)).T
            Bt = B[t, :].reshape((action_dim, state_dim)).T
            ct = c[t, :].reshape((1, state_dim)).T

            if self.predict_deltas:
                next_state = xt + At @ xt + Bt @ ut + ct
            else:
                next_state = At @ xt + Bt @ ut + ct

            opti.subject_to(x[t + 1, :] == next_state.T)

            # Actions Constraints
            for action_constraint in self.action_constraints:
                opti.subject_to(action_constraint(u[t, :]) <= 0)

            # State Constraints
            for state_constraint in self.state_constraints:
                opti.subject_to(state_constraint(x[t, :]) <= 0)

        # Introspection term
        lambd = 0.3
        conservative_obj = 0
        for t in range(0, u.shape[0]):
            step_means = means[t, :].reshape(
                (self.model.k_closest, self.model.n_spatial)
            )
            step_precisions = precisions[t, :].reshape(
                (self.model.k_closest, self.model.n_spatial * self.model.n_spatial)
            )

            ds = []
            for k in range(self.model.k_closest):
                mu = step_means[k, :]
                sigma = step_precisions[k, :].reshape(
                    (self.model.n_spatial, self.model.n_spatial)
                )
                d2 = quad_form(
                    (
                        horzcat(x[t, :], u[t, :])[:, np.array(self.model.spatial_dims)]
                        - mu
                    ).T,
                    sigma,
                )
                ds += [d2]
            ds = vertcat(*ds)
            conservative_obj = conservative_obj - lambd * logsumexp(-d2 / lambd)

        task_objective = cost_fn(x, u, **cost_kwargs)
        objective = (
            self.task_coef * task_objective + self.conservatism_coef * conservative_obj
        )
        opti.minimize(objective)

        problem = opti.to_function(
            "F", [x0, x_prev, u, A, B, c, means, precisions], [x, u, objective]
        )
        self.problem = problem
        self.opti = opti

    def solve_local_QP(self, x0, x_prev, u_prev, return_infos=True):
        # print("Linearize around x_prev:", x_prev[:-1], u_prev)
        (A, B, c), (mean, covariance) = self.linearizer.local_model_many(
            np.hstack((x_prev[:-1], u_prev))
        )
        # print("A", A)
        # print("B", B)
        # print("c", c)
        A_flat = A.reshape(-1, self.state_dim * self.state_dim, order="F")
        B_flat = B.reshape(-1, self.action_dim * self.state_dim, order="F")
        c_flat = c.reshape(-1, 1 * self.state_dim, order="F")

        if self.model.n_agents < self.model.k_closest:
            mean = np.concat([mean] * self.model.k_closest, axis=1)[
                :, : self.model.k_closest
            ]
            covariance = np.concat([covariance] * self.model.k_closest, axis=1)[
                :, : self.model.k_closest
            ]

        mean = mean.reshape(
            self.horizon, self.model.k_closest * self.model.n_spatial, order="F"
        )
        precision = np.linalg.inv(covariance).reshape(
            self.horizon,
            self.model.k_closest * self.model.n_spatial * self.model.n_spatial,
            order="F",
        )

        x, u, cost = self.problem(
            x0, x_prev, u_prev, A_flat, B_flat, c_flat, mean, precision
        )
        if return_infos:
            return (x.toarray(), u.toarray(), cost.toarray()), {
                "A": A,
                "B": B,
                "c": c,
                "cov": covariance,
                "mean": mean,
            }
        return (x.toarray(), u.toarray(), cost.toarray())

    def __call__(self, x: ndarray, return_infos=True):
        x0 = np.array(x[: self.state_dim])
        u_prev = self.u_prev
        if u_prev is None:
            u_prev = np.zeros((self.horizon, self.action_dim))
        u0 = u_prev

        # Initial rollout
        x_prev = self.predictor.predict_trajectory(
            x0, u_prev
        )  # (horizon+1, state_dim) including x0

        max_sqp_iter = self.max_sqp_iter
        alpha_decay = self.alpha_decay
        min_alpha = self.min_alpha
        lin_error_tol = self.lin_error_tol

        infos = {}
        for i in range(max_sqp_iter):
            (x_lin, u_prev, cost), solve_infos = self.solve_local_QP(
                x0, x_prev, u_prev, return_infos=True
            )
            # self.opti.debug.show_infeasibilities()

            # Line Search
            if self.line_search:
                alpha = 1.0
                accepted = False
                while alpha >= min_alpha:
                    u_candidate = u_prev + alpha * (u_prev - u0)
                    x_nl = self.predictor.predict_trajectory(
                        x0, u_candidate
                    )  # Ground Truth
                    lin_error = np.linalg.norm(x_nl - x_lin) / (
                        np.linalg.norm(x_lin) + 1e-6
                    )
                    if lin_error <= lin_error_tol:
                        accepted = True
                        break
                    alpha *= alpha_decay
                if not accepted:
                    break

                u_prev = u_candidate
                x_prev = x_nl
                infos[f"sqp_iter_{i}"] = {
                    "alpha": alpha,
                    "lin_error": lin_error,
                    "cost": cost,
                }

        # Receding horizon shift
        self.u_prev = np.vstack([u_prev[1:], u_prev[-1:]]).clip(-2, 2)

        if return_infos:
            return u_prev, {
                "state_trajectory": x_prev,
                "cost": cost,
                **infos,
                **solve_infos,
            }
        return u_prev
