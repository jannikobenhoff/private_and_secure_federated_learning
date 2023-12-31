import unittest
from src.compressions.SparseGradient import *


class TestSparseGradient(unittest.TestCase):
    def test_gradDrop(self):
        sg = SparseGradient(drop_rate=50)
        grad = tf.random.uniform((40, 50), minval=-3, maxval=3)

        calc = sg.gradDrop(grad, 99)
        calc = tf.cast(calc, dtype=tf.float32)
        print(32 * 2000 / sg.get_sparse_tensor_size_in_bits(calc))
        print(grad, calc)

    def test_gradDrop2d(self):
        sg = SparseGradient(drop_rate=10)
        calc = sg.gradDrop(tf.constant([[0.3, -1.2, 0.9, 0.2], [0.3, 2, 0.9, 0.2]],
                                       dtype=tf.float32), 50)
        print(calc)
        expect = tf.constant([[0, -1.2, 0.9, 0], [0, 2, 0.9, 0]], dtype=tf.float32)

        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
