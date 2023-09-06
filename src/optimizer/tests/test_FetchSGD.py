import unittest
import numpy as np

from src.models.LeNet import LeNet
from src.models.ResNet import ResNet
from src.optimizer.FetchSGD import *


class TestFetchSGD(unittest.TestCase):
    def test_sketch(self):
        gradient = tf.constant([[1, 1, -3, 1, 5], [-1, 2, 1, 4, 5]],
                               dtype=tf.float32)

        f = FetchSGD(c=10, r=1, momentum=0.9, learning_rate=0.1, topk=2)

        # f.build(gradient)

        print(f.local_compress(gradient, gradient, 1))
        print(f.local_compress(gradient, gradient, 1))
        print(f.local_compress(gradient, gradient, 1))

        # gradient = tf.random.uniform((4000, 1))

        # le = LeNet(search=True).search_model(lambda_l2=None)
        # re = ResNet("resnet18", 10, lambda_l2=None)
        # re.build(input_shape=(None, 32, 32, 3))
        # print(le.summary())
        # print(re.summary())

        # re = ResNet("resnet34", 10, lambda_l2=None)
        # re.build(input_shape=(None, 32, 32, 3))
        # print(re.summary())
        # input_shape = gradient.shape
        # d = tf.reshape(gradient, [-1]).shape[0]
        #
        # cs = CSVec(d=d, c=10000, r=1)
        # d = 5000
        # c = 10000
        # r = 1
        # a = CSVec(d, c, r)
        # vec = torch.rand(d)
        #
        # a.accumulateVec(vec)
        #
        # with self.subTest(method="topk"):
        #     recovered = a.unSketch(k=d)
        #     self.assertTrue(torch.allclose(recovered, vec))
        #
        # with self.subTest(method="epsilon"):
        #     thr = vec.abs().min() * 0.9
        #     recovered = a.unSketch(epsilon=thr / vec.norm())
        #     self.assertTrue(torch.allclose(recovered, vec))
        #
        # cs.accumulateVec(tf.reshape(gradient, [-1]).numpy())
        # # Access sketch
        # sketch = cs.table
        # sketch = tf.convert_to_tensor(sketch.numpy(), dtype=gradient.dtype)
        # print("Zeros:", tf.math.count_nonzero(sketch))
        # print(gradient.dtype.size * 8 * np.prod(
        #     gradient.shape.as_list()) / FetchSGD.get_sparse_tensor_size_in_bits(sketch))
        #
        # print(np.prod(gradient.shape.as_list()), np.prod(sketch.shape.as_list()))
        #
        # print("Alt:", gradient.dtype.size * 8 * np.prod(
        #     gradient.shape.as_list()) / (sketch.dtype.size * 8 * np.prod(sketch.shape.as_list())))
        #
        # k = 10000
        # if k > d:
        #     k = d
        # delta = cs.unSketch(k=k)
        # delta = tf.reshape(delta, input_shape)
        #
        # print("diff", tf.math.reduce_sum(tf.abs(delta - gradient)))


if __name__ == '__main__':
    unittest.main()
