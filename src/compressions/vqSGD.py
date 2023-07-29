import numpy as np
import tensorflow as tf
from scipy.optimize import nnls
from tensorflow import Tensor

from src.compressions.Compression import Compression


class vqSGD(Compression):
    def __init__(self, repetition: int, name="vqSGD"):
        super().__init__(name=name)
        self.s = repetition

    def build(self, var_list):
        """Initialize optimizer variables.

        vqSGD optimizer has no variables.

        Args:
          var_list: list of model variables to build vqSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        """
        """
        input_shape = gradient.shape
        l2 = tf.norm(gradient, ord=2)
        gradient = tf.reshape(gradient, [-1]) / l2

        d = gradient.shape[0]
        d_sqrt = np.sqrt(d)

        cp_pos = d_sqrt * tf.eye(d)
        cp = tf.concat([cp_pos, -cp_pos], axis=1)

        a = np.zeros(cp.shape[1])

        gamma = 1 - tf.norm(gradient, ord=1) / d_sqrt
        # for i in range(2 * d):
        #     if gradient[i % d] > 0 and i <= d - 1:
        #         a[i] = gradient[i % d] / d_sqrt + gamma / (2 * d)
        #     elif gradient[i % d] <= 0 and i > d - 1:
        #         a[i] = -gradient[i % d] / d_sqrt + gamma / (2 * d)
        #     else:
        #         a[i] = gamma / (2 * d)
        positive_part = tf.cast(gradient > 0, tf.float32) * ((gradient / d_sqrt) + (gamma / (2 * d)))
        negative_part = tf.cast(gradient <= 0, tf.float32) * ((-gradient / d_sqrt) + (gamma / (2 * d)))

        a[:d] = positive_part

        a[d:] = negative_part
        a = tf.where(a == 0, gamma / (2 * d), a)
        a = tf.cast(a, gradient.dtype)
        print(np.sum(a))

        rand = tf.random.uniform(shape=a.shape)
        a_select = tf.where(a > rand, a, 0)
        compressed_gradient = tf.matmul(cp, tf.reshape(a_select, (-1, 1)))

        return tf.reshape(compressed_gradient, input_shape)
