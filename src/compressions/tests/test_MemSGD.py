import unittest
import numpy as np
from src.optimizer.MemSGD import *


class TestMemSGD(unittest.TestCase):
    def test_top_k(self):
        nc = MemSGD(learning_rate=0.01, top_k=2)
        spars = nc.top_k_sparsification(input_tensor=tf.constant([1, -2.75, 2.75],
                                                                 dtype=tf.float32), k=2)
        print(spars)
        self.assertTrue(tf.math.reduce_all(tf.equal(spars,
                                                    tf.constant([0, -2.75, 2.75], dtype=tf.float32))),
                        "Not equal.")

    def test_top_k_2d(self):
        nc = MemSGD(learning_rate=0.01, top_k=2)
        spars = nc.top_k_sparsification(input_tensor=tf.constant([[1, -2.75, 2.75], [100, 0, 5]],
                                                                 dtype=tf.float32), k=4)
        print(spars)
        self.assertTrue(tf.math.reduce_all(tf.equal(spars,
                                                    tf.constant([[0, -2.75, 2.75],
                                                                 [100, 0, 5]], dtype=tf.float32))),
                        "Not equal.")

    def test_rand_k(self):
        nc = MemSGD(learning_rate=0.01, top_k=2)
        spars = nc.rand_k_sparsification(input_tensor=tf.constant([1, -2.75, 2.75],
                                                                  dtype=tf.float32), k=2)
        print(spars)
        self.assertTrue(spars.shape == (3,),
                        "Not equal.")


if __name__ == '__main__':
    unittest.main()
