import unittest
import numpy as np
from src.optimizer.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_sketch(self):
        # gradient = tf.constant([[1, 1, -3, 1, 5], [-1, 2, 1, 4, 5]],
        #                                dtype=tf.float32)
        gradient = tf.random.uniform((10000, 1))

        input_shape = gradient.shape
        d = tf.reshape(gradient, [-1]).shape[0]

        print(d)
        cs = CSVec(d=d, c=1000, r=100)

        cs.accumulateVec(tf.reshape(gradient, [-1]).numpy())
        # Access sketch
        sketch = cs.table
        k = d
        if k > d:
            k = d
        delta = cs.unSketch(k=k)
        delta = tf.reshape(delta, input_shape)

        print(tf.math.reduce_sum(tf.abs(delta)))
        print(tf.math.reduce_sum(tf.abs(gradient)))
        print("unsketch:", delta-gradient)


if __name__ == '__main__':
    unittest.main()
