import unittest
from src.compressions.SparseGradient import *


class TestSparseGradient(unittest.TestCase):
    def test_gradDrop(self):
        sg = SparseGradient(drop_rate=50)
        grad = tf.constant([0.3, -1.2, 0.9, 0.2])
        calc = sg.gradDrop(grad, 49)
        calc = tf.cast(calc, dtype=tf.float32)
        print(calc)
        get_compression_rate(grad, calc)
        expect = tf.constant([0, -1.2, 0.9, 0], dtype=tf.float32)
        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")

    def test_gradDrop2d(self):
        sg = SparseGradient()
        calc = sg.gradDrop(tf.constant([[0.3, -1.2, 0.9, 0.2], [0.3, 2, 0.9, 0.2]],
                                       dtype=tf.float32), 50)
        print(calc)
        expect = tf.constant([[0, -1.2, 0.9, 0], [0, 2, 0.9, 0]], dtype=tf.float32)

        self.assertTrue(tf.math.reduce_all(tf.equal(calc, expect)) and
                        tf.math.reduce_all(tf.equal(calc.shape, expect.shape)), "Not equal.")


if __name__ == '__main__':
    unittest.main()
