import unittest
from src.compressions.vqSGD import *
from src.utilities.compression_rate import get_compression_rate


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        vq = vqSGD(repetition=5)
        grad = tf.constant([1, 2, 30, -10, 0, 7],
                           dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)


if __name__ == '__main__':
    unittest.main()
