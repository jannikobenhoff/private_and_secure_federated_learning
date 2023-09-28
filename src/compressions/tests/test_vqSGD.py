import unittest
from src.compressions.vqSGD import *


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        # tf.config.set_visible_devices([], 'GPU')

        vq = vqSGD(repetition=10)
        grad = tf.constant([[1, -20, -1], [3, 5.5, 2]], dtype=tf.float32)

        # grad = tf.constant(np.random.randint(high=10, low=-10, size=10), dtype=tf.float32)

        # grad = tf.constant(np.random.uniform(low=-1, high=1, size=(10, 100)),
        #                    dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)
        print(tf.unique_with_counts(tf.abs(calc) > tf.abs(grad))[0])
        print(tf.unique_with_counts(tf.abs(calc) > tf.abs(grad))[2])
        # print(tf.unique_with_counts(tf.sign(calc) * tf.sign(grad))[2])
        print(grad, calc)
        print("Diff:", tf.reduce_sum(tf.abs(grad - calc)).numpy())
        print(tf.norm(grad, ord=1).numpy(), tf.norm(grad, ord=2).numpy())
        print(tf.norm(calc, ord=1).numpy(), tf.norm(calc, ord=2).numpy())

        # print(10000 * np.log2(2 * 795010))


if __name__ == '__main__':
    unittest.main()
