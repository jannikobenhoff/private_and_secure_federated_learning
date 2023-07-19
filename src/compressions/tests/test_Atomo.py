import unittest
from src.compressions.Atomo import *
from src.utilities.compression_rate import get_compression_rate

import torch


def r2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n//2, 2))
    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))
    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0]*shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0]/2), int(tmp_shape[1]*2)))

class TestAtomo(unittest.TestCase):
    def test_compress(self):
        at = Atomo(sparsity_budget=3)
        grad = tf.constant([[[2, 4, 6, 5], [2, 4, 6, 5]]],
                           shape=[4, 2, 1],
                           dtype=tf.float32)

        calc = at.compress(grad, grad)
        print(calc)
        # get_compression_rate(grad, tf.cast(tf.sign(grad), dtype=tf.int8))

        expect = tf.constant([0.4666667, -1.2, 0.4666667, 0.4666667], dtype=tf.float32)
        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")

    def test_reshape(self):
        print(r2d(torch.rand(5, 5, 1, 6)))
        at = Atomo(sparsity_budget=3)
        grad = tf.constant([2, 4, 6, 5],
                           dtype=tf.float32)
        print(at._resize_to_2d(grad))
        grad = tf.constant([[[2, 4, 6, 5], [2, 4, 6, 5]]],
                           shape=[4, 2, 1],
                           dtype=tf.float32)
        print(at._resize_to_2d(grad))


if __name__ == '__main__':
    unittest.main()
