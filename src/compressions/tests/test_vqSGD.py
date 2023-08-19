import unittest
from src.compressions.vqSGD import *


class TestvqSGD(unittest.TestCase):
    def test_compress(self):
        vq = vqSGD(repetition=5)
        grad = tf.constant([[1, 2, 30], [-10, 0, 7]],
                           dtype=tf.float32)

        calc = vq.compress(grad, grad)
        print(calc)

        # input_shape = grad.shape
        # flattened_tensor: Tensor = tf.reshape(grad, [-1])
        # top_k_values, top_k_indices = tf.math.top_k(tf.abs(flattened_tensor), k=4, index_type=tf.int64)
        # top_k_indices = tf.expand_dims(top_k_indices, axis=1)
        #
        # print(input_shape)
        # sparse_tensor = tf.SparseTensor(indices=top_k_indices, values=top_k_values, dense_shape=flattened_tensor.shape)
        # sparse_tensor = tf.reshape(sparse_tensor, shape=input_shape)

        a = tf.abs(grad)
        s = tf.sort(tf.reshape(a, [-1]), direction="DESCENDING")

        bool_mask = tf.reduce_any(tf.equal(tf.reshape(a, [-1, 1]), s[0:3]), axis=-1)
        bool_mask = tf.reshape(bool_mask, tf.shape(grad))

        result = grad * tf.cast(bool_mask, grad.dtype)
        print(result)



if __name__ == '__main__':
    unittest.main()
