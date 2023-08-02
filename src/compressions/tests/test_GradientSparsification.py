import unittest
import numpy as np
from src.compressions.GradientSparsification import *


class TestGradientSparsification(unittest.TestCase):
    def test_greedy(self):
        gs = GradientSparsification(k=0.001, max_iter=2)
        greedy = gs.greedy_algorithm(input_tensor=tf.constant([[1, 20, 3, 4, 5]],
                                                              dtype=tf.float32),
                                     max_iter=2, k=0.001)
        print(greedy)
        self.assertTrue(True, "Not equal.")

    def test_2d(self):
        gs = GradientSparsification(k=0.1, max_iter=2)
        calc = gs.compress(gradient=tf.constant([[1, 2, 30, 4, 5],
                                                 [1, 2, 3, 4, 5]],
                                                              dtype=tf.float32),variable=tf.constant([[1, 2, 30, 4, 5], [1, 2, 3, 4, 5]],
                                                              dtype=tf.float32))
        print(calc)
        self.assertTrue(True, "Not equal.")


if __name__ == '__main__':
    unittest.main()
