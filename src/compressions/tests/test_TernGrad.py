import unittest
import numpy as np
from src.compressions.TernGrad import *


class TestTernGrad(unittest.TestCase):
    def test_ternarize(self):
        tern = TernGrad(clip=2.5)
        grad = tf.constant([[4.3, -1.2, 1.9, 2, 3, 4, 5, 5]], dtype=tf.float32)
        calc = tern.compress(grad, grad)
        print("calc:", calc)

        self.assertTrue(np.array_equal(calc,
                                       tf.constant([0, -1.2, 1.2], dtype=tf.float32)), "Not equal.")

    def test_ternarize_2d(self):
        tern = TernGrad(clip=2.5)
        calc = tern.ternarize(tf.constant([[0.3, -1.2, 0.9], [0.3, -2, 0.9]], dtype=tf.float32))
        expect = tf.constant([[0, -1.2, 1.2], [0, -2, 0]], dtype=tf.float32)
        print(calc)

        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")

    def test_gradient_clipping(self):
        tern = TernGrad(clip=2.5)
        print(tern.gradient_clipping(input_tensor=tf.constant([0.3, -1.2, 0.9], dtype=tf.float32),
                                     c=0.5))


if __name__ == '__main__':
    unittest.main()
