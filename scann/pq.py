
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import jax.scipy.sparse.linalg as jla
from jax import lax, vmap, jit
from functools import partial
import jax.random as jrnd
from scann import VectorQuantizer, anisotropic_weights, loss_function, Quantizer
from scann.logger import Logger


class ProductQuantizer(Quantizer):
    def __init__(self, m, k, T, max_assign_steps=3):
        self.m, self.k, self.T = (m, k, T)
        self.max_assign_steps = max_assign_steps

        self.batch_assign = vmap(self.assign, (None, 0), (0, 0))
        self.batch_build = vmap(self.calculate_B_i, (0,), 0)
        self.batch_calc = vmap(self.calculate_BTx, (0, 0), (0, 0))
    
    def __hash__(self) -> int:
        return hash((
            self.m, self.k, self.T, self.max_assign_steps
        ))

    @partial(jit, static_argnums=0)
    def calculate_B_i(self, I_n):
        indices = []
        values = jnp.array([], dtype=jnp.bool_)

        for i_m in range(self.m):
            i_k = I_n[i_m]
            i_x = i_m * (self.f // self.m)
            i_dk = (i_m * self.f // self.m * self.k) + (i_k * self.f // self.m)

            for i_fm in range(self.f // self.m):
                indices.append(jnp.array([i_x + i_fm, i_dk + i_fm]))
                values = jnp.append(values, 1)
        indices = jnp.array(indices, dtype=jnp.int32)
        B_i = BCOO((values, indices), shape=(self.f, self.f * self.k))
        return B_i

    def assign(self, c, x):
        a_iter = 0
        a_gh = True
        # v-quantized
        q_x = x
        last_l = 0
        while a_gh:
            assignment_row = jnp.zeros((self.m,), dtype=jnp.int16)
            for i_m in range(self.m):
                # select best code word against loss
                k_losses = jnp.array([])
                for i_k in range(self.k):
                    # we want i_m-th codebook and i_k-th codeword
                    i_s = i_k * (self.f // self.m) + (self.f // self.m) * self.k * i_m
                    code_word = lax.dynamic_slice(c, (i_s,), (self.f // self.m,))

                    n_x = lax.dynamic_update_slice(q_x, code_word, (i_m * (self.f // self.m),))
                    l = loss_function(x, n_x, self.weights, normalize=False)
                    k_losses = jnp.append(k_losses, l)
                s_k = jnp.argmin(k_losses)
                si_s = s_k * (self.f // self.m) + (self.f // self.m) * self.k * i_m

                code_word = lax.dynamic_slice(c, (si_s,), (self.f // self.m,))
                q_x = lax.dynamic_update_slice(q_x, code_word, (i_m * (self.f // self.m),))
                # update assignment
                assignment_row = assignment_row.at[i_m].set(s_k)
                last_l = k_losses[s_k]
                pass
            a_iter += 1
            # add check of change
            a_gh = (a_iter < self.max_assign_steps)
        return last_l, assignment_row

    @partial(jit, static_argnums=0)
    def calculate_BTx(self, B_i, x):
        fx = x.reshape((1, x.shape[0]))
        _, h_o, h_p = self.weights
        inner_m = (h_p - h_o) * (fx.T @ fx) + h_o * jnp.eye(x.shape[0])
        lm = B_i.T @ inner_m
        r = lm @ B_i
        return r, h_p * (B_i.T @ fx.T)

    def fit(self, X, tol=1e-2, max_iter=100, vq_impl=True):
        # m -> number of codebooks
        # k -> number of centroids
        data = X
        self.n, self.f = data.shape
        self.weights = anisotropic_weights(self.T, self.f)
        # normalize 
        nX = data / jnp.linalg.norm(data, axis=1, keepdims=True)
        # init 
        # C: k x f      
        Logger.log(self, "Initializing codebook centers ...")
        if vq_impl:
            Logger.log(self, "Using VectorQuantizer to init centers ...")
            vq = VectorQuantizer(self.k, self.T)
            vq.fit(data, tol, max_iter)
            C = vq.C
            self.c = C.reshape(-1)
        else:
            if self.m == 1:
                Logger.log(self, "Choosing centers among data ...")
                key = jrnd.PRNGKey(0)
                key, sub_key = jrnd.split(key)
                C = jrnd.choice(sub_key, nX, shape=(self.k,), replace=False)
                self.c = C.reshape(-1)
            else:
                Logger.log(self, "Using ProductQuantizer to init centers ...")
                pq = ProductQuantizer(1, self.k, self.T, max_assign_steps=1)
                pq.fit(data, tol, max_iter)
                self.c = pq.c
        # int16 ? better'd be determine by k, but need to be efficient too
        self.I = jnp.zeros((self.n, self.m), dtype=jnp.int16)
        # repeat
        go_ahead = True
        iters = 0
        loss = 0

        while go_ahead:
            # assingment step      
            Logger.log(self, "Assigning data points to centers ...")
            a_l, self.I = self.batch_assign(self.c, nX)
            n_loss = jnp.sum(a_l)
            Logger.log(self, "Loss after assignments -> %f", n_loss)
            
            # codeword update step
            # Build B Matrices -> Sparse
            # [i_k, o, i_m]
            # i_x -> pos
            # i_k * i_m * f//m , + f//m
            Logger.log(self, "Creating B Matrices ...")
            B = self.batch_build(self.I)

            # Shape hints
            # BTB = jnp.zeros((self.f * self.k, self.f * self.k))
            # BTx = jnp.zeros((self.f * self.k, 1))
            Logger.log(self, "Computing BTB, BTx Matrices ...")
            # TODO: Memory Optimization
            BTBb, BTxb = self.batch_calc(B, nX)
            BTB = BTBb.sum(axis=0)
            BTx = BTxb.sum(axis=0)
            
            # update centers
            Logger.log(self, "Finding centers using conjugate gradient ...")
            c, _ = jla.cg(BTB, BTx)
            self.c = c.reshape(-1)
            
            iters += 1
            go_ahead = (abs(n_loss - loss) > tol) and (iters < max_iter)
            loss = n_loss

    def quantize(self, x):
        return self.assign(self.c, x)
