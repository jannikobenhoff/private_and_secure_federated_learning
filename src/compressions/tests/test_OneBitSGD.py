import unittest
from src.compressions.OneBitSGD import *


class TestOneBitSGD(unittest.TestCase):
    def test_unqantize(self):
        ob = OneBitSGD()
        grad = tf.constant([0.3, -1.2, 0.9, 0.2],
                           dtype=tf.float32)

        calc = tf.cast(ob.unquantize(tf.sign(grad), grad), dtype=tf.float32)
        print(calc)
        print(tf.cast(tf.sign(grad), dtype=tf.int8))

        expect = tf.constant([0.4666667, -1.2, 0.4666667, 0.4666667], dtype=tf.float32)
        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
