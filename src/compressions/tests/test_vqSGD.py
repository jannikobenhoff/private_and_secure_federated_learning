import unittest
from src.compressions.vqSGD import *


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        vq = vqSGD(repetition=5)
        grad = tf.constant([[0, 0, 1], [0, 0, 2]],
                           dtype=tf.float32)
        # grad = tf.constant(np.random.uniform(low=-1, high=1, size=(10, 100)),
        #                    dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)
        print(tf.reduce_sum(tf.abs(grad - calc)).numpy())
        print(tf.norm(grad, ord=1).numpy(), tf.norm(grad, ord=2).numpy())
        print(tf.norm(calc, ord=1).numpy(), tf.norm(calc, ord=2).numpy())

        # direction gleich???


if __name__ == '__main__':
    unittest.main()
