import jax.scipy.sparse.linalg as jla
import jax
import jax.numpy as jnp
import jax.random as jrnd
from scann import loss_function, anisotropic_weights, loss_function_batch
import jax.experimental.host_callback as hcb


def find_center(x, C, weights, normalize=True):
    losses = jax.lax.map(lambda c: loss_function(x, c, weights, normalize=normalize), C)
    # in case of need fore debugging in jaxpr functions (jax.map, jax.jit, ...)
    # hcb.id_print(losses)
    i = jnp.argmin(losses)
    return i


def optimize_center(X_j, weights):
    _, h_o, h_p = weights
    _, f = X_j.shape
    A = (jnp.eye(f) * len(X_j) * h_o)
    A += (h_p - h_o) * (X_j.T @ X_j)

    b = h_p * jnp.sum(X_j, axis=0)
    # Solve Ax = b
    nc, _ = jla.cg(A, b)
    return nc

def vector_quantize(data, k, T, tol=1e-2, max_iter=100):
    n, f = data.shape
    eta, h_o, h_p = anisotropic_weights(T, f)
    # normalize 
    nX = data / jnp.linalg.norm(data, axis=1, keepdims=True)
    # init 
    # C: k x f
    key = jrnd.PRNGKey(0)
    key, sub_key = jrnd.split(key)
    C = jrnd.choice(sub_key, nX, shape=(k,), replace=False)
    # repeat
    go_ahead = True
    iters = 0
    loss = 0
    while go_ahead:
        # partition assignment
        I_c = jax.lax.map(lambda x: find_center(x, C, weights=(eta, h_o, h_p), normalize=False), nX)
        # loss
        X_q = jax.lax.map(lambda i: C[i, :], I_c)
        n_loss = loss_function_batch(nX, X_q, (eta, h_o, h_p), normalize=False)
        print("loss ->", n_loss)
        # codebook optimization
        
        for j in range(k):
            X_j = nX[I_c == j]
            C = C.at[j].set(optimize_center(X_j, (eta, h_o, h_p)))

        iters += 1
        go_ahead = (abs(n_loss - loss) > tol) and (iters < max_iter)
        loss = n_loss
    return C