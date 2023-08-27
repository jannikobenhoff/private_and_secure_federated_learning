import unittest
from src.compressions.Atomo import *

import torch


def r2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n // 2, 2))
    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))
    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0] * shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0] / 2), int(tmp_shape[1] * 2)))


class TestAtomo(unittest.TestCase):
    def test_compress(self):
        at = Atomo(svd_rank=1)
        # grad = tf.constant([[2, 4, 6, 5], [2, 4, 6, 5]],
        #                    dtype=tf.float32)

        grad = tf.constant(np.random.rand(1000, 100), shape=[1000, 20, 5], dtype=tf.float32)

        calc = at.compress(grad, grad)
        print("Diff:", tf.reduce_sum(tf.abs(calc - grad)))


if __name__ == '__main__':
    unittest.main()
