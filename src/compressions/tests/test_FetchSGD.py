import unittest
import numpy as np
from src.optimizer.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_top_k(self):
        fs = FetchSGD(learning_rate=0.01, momentum=0.9)
        sketch = fs.count_sketch(input_tensor=tf.constant([1, 2, 3, 4, 5],
                                                          dtype=tf.float32),
                                 num_counters=2, num_hash_functions=3)
        print(sketch)
        print(fs.uncount_sketch(sketch, 2, 3))
        self.assertTrue(sketch.shape == (), "Not equal.")


if __name__ == '__main__':
    unittest.main()
