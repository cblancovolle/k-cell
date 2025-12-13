from cell.agents.linear_agent import LinearAgent
from cell.trainers.online_trainer import OnlineTrainer
from mpcell.common.constraints import LinearConstraint
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
    ):
        assert model.agent_cls in [LinearAgent]
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.linearizer: StateActionLinearizerWrapper = StateActionLinearizerWrapper(
            model, state_dim, action_dim
        )
        self.predictor: StateActionPredictorWrapper = StateActionPredictorWrapper(model)
        self.horizon = horizon

        self.state_constraints: list[LinearConstraint] = state_constraints
        self.action_constraints: list[LinearConstraint] = action_constraints

    def reset(self):
        self.u_prev = None

    def build_problem(self, cost_fn, cost_kwargs):
        horizon = self.horizon
        state_dim = self.state_dim
        action_dim = self.action_dim

        opti = Opti("conic")
        opti.solver(
            "osqp",
            {
                "error_on_fail": False,
                # "warm_start_dual": False,
                # "warm_start_primal": False,
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
            next_state = xt + At @ xt + Bt @ ut + ct
            opti.subject_to(x[t + 1, :] == next_state.T)

            # Actions Constraints
            for action_constraint in self.action_constraints:
                opti.subject_to(action_constraint(u[t, :]) <= 0)

            # State Constraints
            for state_constraint in self.state_constraints:
                opti.subject_to(state_constraint(x[t, :]) <= 0)

        task_objective = cost_fn(x, u, **cost_kwargs)
        objective = task_objective
        opti.minimize(objective)

        problem = opti.to_function("F", [x0, x_prev, u, A, B, c], [x, u, objective])
        self.problem = problem

    def solve_local_QP(self, x0, x_prev, u_prev, return_infos=True):
        (A, B, c), (mean, covariance) = self.linearizer.local_model_many(
            np.hstack((x_prev[:-1], u_prev))
        )
        A_flat = A.reshape(-1, self.state_dim * self.state_dim, order="F")
        B_flat = B.reshape(-1, self.action_dim * self.state_dim, order="F")
        c_flat = c.reshape(-1, 1 * self.state_dim, order="F")

        x, u, cost = self.problem(x0, x_prev, u_prev, A_flat, B_flat, c_flat)
        if return_infos:
            return (x.toarray(), u.toarray(), cost.toarray()), {
                "A": A,
                "B": B,
                "c": c,
                "cov": covariance,
                "mean": mean,
            }
        return (x.toarray(), u.toarray(), cost.toarray())
