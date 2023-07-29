import unittest
import numpy as np
from src.optimizer.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_top_k(self):
        fs = FetchSGD(learning_rate=0.01, sketch_size=2)
        # sketch = fs.count_sketch(tf.constant([[1, 1, -3, 1, 5], [-1, 2, 1, 4, 5]],
        #                                                   dtype=tf.float32),
        #                          3)
        # print("sketch:", sketch)
        # print(fs.uncount_sketch(sketch, 3))
        sketch = fs.sketch(tf.constant([[1, 1, -3, 1, 5], [-1, 2, 1, 4, 5]],
                                       dtype=tf.float32))
        print("sketch:", sketch)


if __name__ == '__main__':
    unittest.main()
