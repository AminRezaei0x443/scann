import unittest

import jax.random as jrnd
from scann import vector_quantize, product_quantize
from .util import timeit


class QuantizationTesting(unittest.TestCase):
    def setUp(self):
        k = jrnd.PRNGKey(0)
        self.data = jrnd.uniform(k, shape=(25, 64))

    @timeit
    def test_vq(self):
        vector_quantize(self.data, 4, 0.2, tol=1e-4)
        pass

    @timeit
    def test_pq(self):
        product_quantize(self.data, 4, 10, 0.2)
        pass