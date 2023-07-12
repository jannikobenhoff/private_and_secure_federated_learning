import unittest
import numpy as np
from src.compressions.TernGrad import *


class TestTernGrad(unittest.TestCase):
    def test_ternarize(self):
        tern = TernGrad(learning_rate=0.01)
        calc = tern.ternarize(tf.constant([0.3, -1.2, 0.9], dtype=tf.float32))
        print(calc)
        self.assertTrue(np.array_equal(calc,
                                       tf.constant([0, -1.2, 1.2], dtype=tf.float32)), "Not equal.")

    def test_ternarize_2d(self):
        tern = TernGrad(learning_rate=0.01)
        calc = tern.ternarize(tf.constant([[0.3, -1.2, 0.9], [0.3, -1.2, 0.9]], dtype=tf.float32))
        expect = tf.constant([[0, -1.2, 1.2], [0, -1.2, 1.2]], dtype=tf.float32)
        print(calc)

        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")

    def test_ternarize_3d(self):
        tern = TernGrad(learning_rate=0.01)
        calc = tern.ternarize(tf.constant([[[0.3, -1.2, 0.9, 0.3],
                                            [0.3, 0.3, 0.3, 0.3]],
                                           [[0.3, 0.3, 0.3, 0.3],
                                            [0.3, 0.3, 0.3, 0.3]],
                                           [[0.3, 0.3, 0.3, 0.3],
                                            [0.3, 0.3, 0.3, 0.3]]], dtype=tf.float32))
        expect = tf.constant([[[0., -1.2, 1.2, 0.],
                                            [0., 0., 0., 0.]],
                                           [[0., 0., 0., 0.],
                                            [0., 0, 0., 0.]],
                                           [[0., 0, 0., 0.],
                                            [0., 0, 0., 0.]]], dtype=tf.float32)
        print(calc)

        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
