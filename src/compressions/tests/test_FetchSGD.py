import unittest
import numpy as np
from src.compressions.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_top_k(self):
        fs = FetchSGD(learning_rate=0.01, momentum=1)
        sketch = fs.count_sketch(input_tensor=tf.constant([1, 2, 3, 4, 5],
                                                          dtype=tf.float32),
                                 num_counters=5, num_hash_functions=3)
        print(sketch)
        self.assertTrue(sketch.shape == (), "Not equal.")


if __name__ == '__main__':
    unittest.main()
