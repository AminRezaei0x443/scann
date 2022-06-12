import unittest

import jax.random as jrnd
import jax.numpy as jnp
from scann import VectorQuantizer, ProductQuantizer
from test.util import timeit


class QuantizationTesting(unittest.TestCase):
    def setUp(self):
        k = jrnd.PRNGKey(0)
        self.data = jrnd.uniform(k, shape=(25, 64))

    @timeit
    def test_vq(self):
        vq = VectorQuantizer(4, 0.2)
        vq.fit(self.data, tol=1e-4)
        k = jrnd.PRNGKey(34)
        x = jrnd.uniform(k, shape=(64,))
        x /= jnp.dot(x, x)
        print(vq.quantize(x))

    @timeit
    def test_pq(self):
        pq = ProductQuantizer(4, 10, 0.2)
        pq.fit(self.data)
        k = jrnd.PRNGKey(34)
        x = jrnd.uniform(k, shape=(64, ))
        x /= jnp.dot(x, x)
        print(pq.quantize(x))

    @timeit
    def test_pq_large(self):
        k = jrnd.PRNGKey(0)
        data = jrnd.uniform(k, shape=(500, 768))
        pq = ProductQuantizer(16, 16, 0.2)
        pq.fit(data)
