import unittest
from src.compressions.OneBitSGD import *


class TestOneBitSGD(unittest.TestCase):
    def test_unqantize(self):
        tf.config.run_functions_eagerly(run_eagerly=True)
        tf.data.experimental.enable_debug_mode()

        ob = OneBitSGD()
        grad = tf.constant([3, -1, 2, -2], name="a",
                           dtype=tf.float32)

        calc = ob.compress([grad], [grad])
        print(calc)
        # {'compressed_grads': [<tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 1., -1.,  1., -1.], dtype=float32)>], 'decompress_info': [(-1.5, 2.5)], 'needs_decompress': True}


if __name__ == '__main__':
    unittest.main()
