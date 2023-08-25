import unittest
import numpy as np

from src.models.LeNet import LeNet
from src.models.ResNet import ResNet
from src.optimizer.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_sketch(self):
        # gradient = tf.constant([[1, 1, -3, 1, 5], [-1, 2, 1, 4, 5]],
        #                                dtype=tf.float32)
        gradient = tf.random.uniform((4000000, 1))

        le = LeNet(search=True).search_model(lambda_l2=None)
        re = ResNet("resnet18", 10, lambda_l2=None)
        re.build(input_shape=(None, 32, 32, 3))
        print(le.summary())
        # print(re.summary())
        re = ResNet("resnet34", 10, lambda_l2=None)
        re.build(input_shape=(None, 32, 32, 3))
        # print(re.summary())
        input_shape = gradient.shape
        d = tf.reshape(gradient, [-1]).shape[0]

        cs = CSVec(d=d, c=1000000, r=1)

        cs.accumulateVec(tf.reshape(gradient, [-1]).numpy())
        # Access sketch
        sketch = cs.table
        sketch = tf.convert_to_tensor(sketch.numpy(), dtype=gradient.dtype)

        print(gradient.dtype.size * 8 * np.prod(
            gradient.shape.as_list()) / FetchSGD.get_sparse_tensor_size_in_bits(sketch))

        print(np.prod(gradient.shape.as_list()), np.prod(sketch.shape.as_list()))

        print("Alt:", gradient.dtype.size * 8 * np.prod(
            gradient.shape.as_list()) / (sketch.dtype.size * 8 * np.prod(sketch.shape.as_list())))

        k = 5000
        if k > d:
            k = d
        delta = cs.unSketch(k=k)
        delta = tf.reshape(delta, input_shape)
        print("topk", gradient.dtype.size * 8 * np.prod(
            gradient.shape.as_list()) / FetchSGD.get_sparse_tensor_size_in_bits(
            delta))
        print(tf.math.reduce_sum(tf.abs(delta)))
        print(tf.math.reduce_sum(tf.abs(gradient)))
        print("unsketch:", delta - gradient)


if __name__ == '__main__':
    unittest.main()
