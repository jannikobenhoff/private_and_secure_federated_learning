import unittest
from src.compressions.vqSGD import *


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        vq = vqSGD(repetition=1)
        # grad = tf.constant([[10, -20, -11], [3, 5.5, 20]], dtype=tf.float32)

        grad = tf.constant(np.random.rand(100, 100), dtype=tf.float32)

        # grad = tf.constant(np.random.uniform(low=-1, high=1, size=(10, 100)),
        #                    dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)
        print("Diff:", tf.reduce_sum(tf.abs(grad - calc)).numpy())
        print(tf.norm(grad, ord=1).numpy(), tf.norm(grad, ord=2).numpy())
        print(tf.norm(calc, ord=1).numpy(), tf.norm(calc, ord=2).numpy())

        # print(10000 * np.log2(2 * 795010))


if __name__ == '__main__':
    unittest.main()
