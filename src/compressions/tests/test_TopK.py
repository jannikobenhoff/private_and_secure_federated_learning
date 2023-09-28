import unittest
from src.compressions.TopK import *
import numpy as np


class TestTopK(unittest.TestCase):
    def test_topk(self):
        topk = TopK(k=10)
        grad = tf.constant([1, 2, 3, 4, 5, -6, 7, -8, 9, 10], dtype=tf.float32)
        print(topk.top_k_sparsification(input_tensor=grad, k=9))


if __name__ == '__main__':
    unittest.main()
