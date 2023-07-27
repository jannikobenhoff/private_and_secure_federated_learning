import unittest
from src.compressions.vqSGD import *
from src.utilities.compression_rate import get_compression_rate


class TestvqSGD(unittest.TestCase):
    def test_probabilites(self):
        vq = vqSGD()
        grad = tf.constant([[[2, 4, 6, 5], [2, 4, 6, 5]]],
                           shape=[4, 2, 1],
                           dtype=tf.float32)

        calc = vq.probabilities(grad)
        print(calc)
        # get_compression_rate(grad, tf.cast(tf.sign(grad), dtype=tf.int8))

        expect = tf.constant([0.4666667, -1.2, 0.4666667, 0.4666667], dtype=tf.float32)
        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
