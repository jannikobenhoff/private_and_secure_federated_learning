import unittest
import numpy as np
from src.compressions.NaturalCompression import *
from src.utilities.compression_rate import get_compression_rate


class TestNaturalCompression(unittest.TestCase):
    def test_compression(self):
        nc = NaturalCompression()
        grad = tf.constant([0, -2.75, 2.75], dtype=tf.float32)
        c_nat = nc.compress(gradient=grad, variable=0)
        print(c_nat)
        get_compression_rate(grad, c_nat)

    def test_compression_2d(self):
        nc = NaturalCompression()
        c_nat = nc.compress(gradient=tf.constant([[1, -2.75, 3], [4, -5, 8]], dtype=tf.float32), variable=None)
        print(c_nat)

    def test_compression_3d(self):
        nc = NaturalCompression()
        calc = nc.compress(gradient=tf.ones([3, 2, 4, 5])*-2.75, variable=None)
        print(calc)


if __name__ == '__main__':
    unittest.main()
