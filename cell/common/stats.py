import torch


def update_covariance_welford(cov, mu, n, x):
    delta = x - mu
    mu_new = mu + delta / (n + 1)
    delta2 = x - mu_new

    C = cov * (n - 1)
    C_new = C + torch.outer(delta, delta2)
    return C_new / n, mu_new


def mahalanobis2(x, P, mu):
    """Compute squared mahalanobis distance

    Args:
        x (Tensor): (n_dim,)
        P (Tensor): precision matrix (n_dim, n_dim)
        mu (Tensor): mean (state_dim,)

    Returns:
        Tensor: (1,)
    """
    return (x - mu).T @ P @ (x - mu)
