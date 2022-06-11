import unittest

import jax.random as jrnd
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

    @timeit
    def test_pq(self):
        pq = ProductQuantizer(4, 10, 0.2)
        pq.fit(self.data)
