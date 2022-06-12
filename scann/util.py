import jax.numpy as jnp


def compute_eta(T, n_dim):
    return (n_dim-1) * (T ** 2) / (1 - T ** 2)

def anisotropic_weights(T, n_dim):
    eta = compute_eta(T, n_dim)
    h_orth = 1 / (1 + eta)
    h_par = 1 - h_orth
    return eta, h_orth, h_par

def loss_function(x, x_q, weights, normalize=False):
    _, h_o, h_p = weights
    d = x - x_q
    r_p = jnp.dot(d, x) * x 
    if normalize:
        r_p /= jnp.dot(x, x)
    r_o = d - r_p
    l = h_p * jnp.dot(r_p, r_p) + h_o * jnp.dot(r_o, r_o)
    return l

def loss_function_batch(X, Xq, weights, normalize=False):
    _, h_o, h_p = weights
    n, _ = X.shape
    D = X - Xq
    # parallel residual
    r_p = jnp.sum(D * X, axis=1).reshape((n, 1)) * X 
    if normalize:
        r_p /= jnp.linalg.norm(X, axis=1, keepdims=True)
    r_o = D - r_p
    l = h_p * jnp.linalg.norm(r_p, axis=1) + h_o * jnp.linalg.norm(r_o, axis=1)
    l = jnp.sum(l)
    return l