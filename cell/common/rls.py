import torch
from torch import Tensor


def fit_linear_regression(X, y, weights, l1_penalty=1e-3):
    """Perform a weighted linear regression

    Args:
        X (Tensor): (batch_size, input_dim)
        y (Tensor): (batch_size, output_dim)
        weights (Tensor): (batch_size, 1)

    Returns:
        Tensor: (input_dim + 1, output_dim)
    """
    num_samples = X.size(0)
    X_with_bias = torch.cat((X, torch.ones((num_samples, 1), device=X.device)), dim=-1)
    W = torch.diag(weights)
    X_weighted = W @ X_with_bias
    y_weighted = W @ y
    # Concatenate a column of ones to X for the bias term
    # Compute parameters
    identity_matrix = torch.eye(X_weighted.size(1), device=X.device)
    identity_matrix[-1, -1] = 0.0
    parameters = torch.linalg.lstsq(
        X_weighted.T @ X_weighted + l1_penalty * identity_matrix,
        X_weighted.T @ y_weighted,
        driver="gels",  # /!\ ULTRA IMPORTANT for numpy equivalence (and GPU/CPU equivalence too)
    ).solution

    return parameters


def rls_init(X: Tensor, y: Tensor, jitter=1e-8):
    """
    Initialize theta and P from a batch in PyTorch

    Args:
        X (Tensor): (n_samples, n_features)
        y (Tensor): (n_samples,)
        ridge (float, optional): jitter for invertibility. Defaults to 1e-8.

    Returns:
        tuple[Tensor, Tensor]: theta0 (n_features+1, 1), P0 (n_features+1, n_features+1)
    """
    n, d = X.shape
    X_with_bias = torch.cat([X, torch.ones((n, 1), dtype=X.dtype)], dim=1)

    W = torch.eye(n, dtype=X.dtype)

    theta0 = fit_linear_regression(X, y, weights=torch.ones(n, dtype=X.dtype))
    P0 = torch.linalg.inv(
        X_with_bias.T @ W @ X_with_bias + jitter * torch.eye(d + 1, dtype=X.dtype)
    )

    return theta0, P0


def mimo_rls_init(X: Tensor, y: Tensor, jitter=1e-8):
    theta0, P0 = torch.vmap(rls_init, in_dims=(None, 1), out_dims=(1, 0))(X, y)
    return theta0, P0


def rls_update(theta: Tensor, P: Tensor, x: Tensor, y: Tensor, forgetting=0.99):
    """
    RLS update with bias

    Args:
        theta (Tensor): (n_features+1, 1)
        P (Tensor): (n_features+1, n_features+1)
        x (Tensor): (n_features,)
        y (Tensor): (1,)
        forgetting (float, optional): forgetting factor. Defaults to 0.99.

    Returns:
        tuple[Tensor, Tensor]: theta_new (n_features+1, 1), P_new (n_features+1, n_features+1)
    """
    x = torch.cat([x, torch.ones(1, dtype=x.dtype)]).view(-1, 1)  # for intercept
    Px = P @ x  # (n_features + 1)
    error = y.view(1, -1) - (x.T @ theta)
    K = Px / (forgetting + (x.T @ Px))
    theta_new = theta + K @ error
    P_new = (P - K @ (x.T @ P)) / forgetting

    return theta_new, P_new


def mimo_rls_update(theta: Tensor, P: Tensor, x: Tensor, y: Tensor, forgetting=0.99):
    new_theta, new_P = torch.vmap(rls_update, in_dims=(1, 0, None, 1), out_dims=(1, 0))(
        theta, P, x, y.view(1, -1), forgetting=forgetting
    )
    return new_theta, new_P


def rls_predict(theta, x):
    x = torch.cat([x, torch.ones(1, dtype=x.dtype)]).view(-1, 1)
    return x.T @ theta
