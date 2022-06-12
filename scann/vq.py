from functools import partial
import jax.scipy.sparse.linalg as jla
import jax
import jax.numpy as jnp
import jax.random as jrnd
from scann import loss_function, anisotropic_weights, Quantizer
from scann.logger import Logger
import jax.experimental.host_callback as hcb


class VectorQuantizer(Quantizer):
    def __init__(self, k, T):
        self.k = k
        self.T = T
        # compile functions
        self.batch_optimize_center = jax.jit(jax.vmap(self.optimize_center, (0, 0), 0))
        self.batch_find_center = jax.jit(jax.vmap(self.find_center, (0, None), (0, 0)))
        self.batch_loss = jax.jit(jax.vmap(loss_function, (None, 0, None), 0))

    def optimize_center(self, X_j, nz):
        _, h_o, h_p = self.weights
        _, f = X_j.shape
        A = (jnp.eye(f) * nz * h_o)
        A += (h_p - h_o) * (X_j.T @ X_j)

        b = h_p * jnp.sum(X_j, axis=0)
        # Solve Ax = b
        nc, _ = jla.cg(A, b)
        return nc 

    def find_center(self, x, C):
        losses = self.batch_loss(x, C, self.weights)
        # in case of need fore debugging in jaxpr functions (jax.map, jax.jit, ...)
        # hcb.id_print(losses)
        i = jnp.argmin(losses)
        return i, losses[i]

    def fit(self, X, tol=1e-2, max_iter=100):
        data = X
        self.n, self.f = data.shape
        self.weights = anisotropic_weights(self.T, self.f)
        # normalize 
        nX = data / jnp.linalg.norm(data, axis=1, keepdims=True)
        # init 
        # C: k x f
        key = jrnd.PRNGKey(0)
        key, sub_key = jrnd.split(key)
        C = jrnd.choice(sub_key, nX, shape=(self.k,), replace=False)
        # repeat
        go_ahead = True
        iters = 0
        loss = 0
        while go_ahead:
            # partition assignment
            Logger.log(self, "Assigning data points to centers ...")
            I_c, losses = self.batch_find_center(nX, C)
            # loss
            n_loss = losses.sum()
            Logger.log(self, "Loss after assignments -> %f", n_loss)
            Logger.log(self, "Optimizing centers ...")
            # codebook optimization
            masks = []
            nzs = []
            for j in range(self.k):
                mask = jnp.repeat((I_c == j).reshape(self.n, 1), self.f, axis=1)
                X_j = nX * mask
                masks.append(X_j)
                nzs.append((I_c == j).sum())

            masks = jnp.stack(masks)
            nzs = jnp.stack(nzs)
            C = self.batch_optimize_center(masks, nzs)

            iters += 1
            go_ahead = (abs(n_loss - loss) > tol) and (iters < max_iter)
            loss = n_loss
        self.C = C

    def quantize(self, x):
        return super().quantize(x)