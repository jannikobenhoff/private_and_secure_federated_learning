import unittest
import numpy as np
from src.compressions.NaturalCompression import *


class TestNaturalCompression(unittest.TestCase):
    def test_compression(self):
        nc = NaturalCompression(learning_rate=0.01)
        c_nat = nc.compress(input_tensor=tf.constant([0, -2.75, 2.75], dtype=tf.float32))
        print(c_nat)
        self.assertTrue(tf.math.reduce_all(tf.equal(c_nat,
                                                    tf.constant([0, -2, 2], dtype=tf.float32))),
                        "Not equal.")

    def test_compression_2d(self):
        nc = NaturalCompression(learning_rate=0.01)
        c_nat = nc.compress(input_tensor=tf.constant([[1, -2.75, 3], [4, -5, 8]], dtype=tf.float32))
        print(c_nat)
        self.assertTrue(tf.math.reduce_all(tf.equal(c_nat,
                                                    tf.constant([[1, -2, 2], [4, -4, 8]], dtype=tf.float32))),
                        "Not equal.")

    def test_compression_3d(self):
        nc = NaturalCompression(learning_rate=0.01)
        calc = nc.compress(input_tensor=tf.ones([3, 2, 4, 5])*-2.75)
        expect = tf.ones([3, 2, 4, 5])*-2
        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
