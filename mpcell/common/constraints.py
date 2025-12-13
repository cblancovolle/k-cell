from casadi import *


def leq(A, b):
    """Generate a linear constraint inequality function

    Args:
        A (ndarray): (n_constraints, n_dim)
        b (ndarray): (n_constraints,)

    Returns:
        callable: constraint function
    """
    x = MX.sym("x", A.shape[1])
    return Function(
        "c_leq", [x], [(lambda x: MX(A) @ x.reshape((A.shape[1], -1)) - MX(b))(x)]
    )


def geq(A, b):
    x = MX.sym("x", A.shape[1])
    return Function("c_geq", [x], [(lambda x: -leq(A, b)(x))(x)])


class Constraint:
    def __call__(self, x):
        return NotImplementedError


class LinearConstraint(Constraint):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def __call__(self, x):
        return leq(self.A, self.b)(x)

    def __repr__(self):
        return f"{self.A} @ x <= {self.b}"


def linearize_constraint(g, x, x0):
    """
    Linearizes a CasADi expression g(x) around point x0.

    Parameters
    ----------
    g : casadi.MX or casadi.SX
        Expression to linearize (scalar or vector).
    x : casadi.MX or casadi.SX
        Decision variable(s).
    x0 : casadi.MX, casadi.SX, or Opti parameter
        Expansion point around which to linearize.

    Returns
    -------
    g_lin : casadi.MX or casadi.SX
        Symbolic linearized expression: g_lin ≈ g(x0) + Jg(x0) * (x - x0)
    """
    # Compute g at x0 (symbolic substitute)
    g0 = substitute(g, x, x0)

    # Compute Jacobian w.r.t x
    Jg = jacobian(g, x)

    # Evaluate Jacobian at x0
    Jg0 = substitute(Jg, x, x0)

    # Linearized expression
    g_lin = g0 + mtimes(Jg0, (x - x0))

    return g_lin
