
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import jax.scipy.sparse.linalg as jla
from scann import vector_quantize, anisotropic_weights, loss_function


def product_quantize(data, m, k, T, tol=1e-2, max_iter=100):
    # m -> number of codebooks
    # k -> number of centroids
    n, f = data.shape
    eta, h_o, h_p = anisotropic_weights(T, f)
    # normalize 
    nX = data / jnp.linalg.norm(data, axis=1, keepdims=True)
    # init 
    # C: k x f
    C = vector_quantize(data, k, T, tol, max_iter)
    c = C.reshape(-1)
    # int16 ? better'd be determine by k, but need to be efficient too
    I = jnp.zeros((n, m), dtype=jnp.int16)
    # repeat
    go_ahead = True
    iters = 0
    loss = 0

    while go_ahead:
        # assingment step
        n_loss = 0
        
        print("assigning ...")
        max_assign_steps = 3
        for i_n, x in enumerate(nX):
            a_iter = 0
            a_gh = True
            # v-quantized
            q_x = x
            last_l = 0
            while a_gh:
                for i_m in range(m):
                    # select best code word against loss
                    k_losses = []
                    for i_k in range(k):
                        # we want i_m-th codebook and i_k-th codeword
                        i_s = i_k * (f // m) + (f // m) * k * i_m
                        i_e = i_s + (f // m)
                        code_word = c[i_s:i_e]

                        n_x = q_x.at[i_m * (f // m) : (i_m + 1) * (f // m)].set(code_word)
                        l = loss_function(x, n_x, (eta, h_o, h_p), normalize=False)
                        k_losses.append(l)
                    s_k = jnp.argmin(jnp.array(k_losses))
                    si_s = s_k * (f // m) + (f // m) * k * i_m
                    si_e = si_s + (f // m)
                    q_x = q_x.at[i_m * (f // m) : (i_m + 1) * (f // m)].set(c[si_s:si_e])
                    # update assignment
                    I = I.at[i_n, i_m].set(s_k)
                    last_l = k_losses[s_k]
                    pass
                a_iter += 1
                # add check of change
                a_gh = (a_iter < max_assign_steps)
            n_loss += last_l
        
        # codeword update step
        print("loss after assigns ->", n_loss)
        
        # Build B Matrices -> Sparse
        # [i_k, o, i_m]
        # i_x -> pos
        # i_k * i_m * f//m , + f//m
        B = []
        print("creating B_i ...")
        for i_n, x in enumerate(nX):
            # we need indices , values
            indices = []
            values = jnp.array([], dtype=jnp.bool_)
            for i_m in range(m):
                i_k = I[i_n, i_m]
                i_x = i_m * (f//m)
                i_dk = (i_m * f//m * k) + (i_k * f//m)
     
                for i_fm in range(f//m):
                    indices.append(jnp.array([i_x + i_fm, i_dk + i_fm]))
                    values = jnp.append(values, 1)
            indices = jnp.array(indices, dtype=jnp.int32)
            B_i = BCOO((values, indices), shape=(f, f * k))
            B.append(B_i)

        BTB = jnp.zeros((f*k, f*k))
        BTx = jnp.zeros((f*k, 1))
        print("calculating BTB, BTx ...")
        for i_n, x in enumerate(nX):
            fx = x.reshape((1, x.shape[0]))
            inner_m = (h_p - h_o) * (fx.T @ fx) + h_o * jnp.eye(x.shape[0])
            B_i = B[i_n]
            lm = B_i.T @ inner_m
            r = lm @ B_i

            BTB += r
            BTx += h_p * (B_i.T @ fx.T)
        
        # update centers
        print("cg ...")
        c, _ = jla.cg(BTB, BTx)
        c = c.reshape(-1)
        
        iters += 1
        go_ahead = (abs(n_loss - loss) > tol) and (iters < max_iter)
        loss = n_loss
    return c

