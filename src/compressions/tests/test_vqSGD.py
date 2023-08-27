import unittest
from src.compressions.vqSGD import *


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        vq = vqSGD(repetition=5)
        grad = tf.constant([[0, 0, -11], [0, 0, 20]], dtype=tf.float32)

        # grad = tf.constant(np.random.rand(1000, 100), shape=[1000, 20, 5], dtype=tf.float32)

        # grad = tf.constant(np.random.uniform(low=-1, high=1, size=(10, 100)),
        #                    dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)
        print("Sum:", tf.reduce_sum(tf.abs(grad - calc)).numpy())
        print(tf.norm(grad, ord=1).numpy(), tf.norm(grad, ord=2).numpy())
        print(tf.norm(calc, ord=1).numpy(), tf.norm(calc, ord=2).numpy())

        # direction gleich???

        # print(10000 * np.log2(2 * 795010))
        d = 10
        indices = np.array([1, 2, 3, 4, 7, 12, 14])
        compressed_gradient = np.zeros(10)
        compressed_gradient[indices[indices < d]] += 1
        compressed_gradient[indices[indices >= d] - d] -= 1

        print(compressed_gradient)


if __name__ == '__main__':
    unittest.main()
