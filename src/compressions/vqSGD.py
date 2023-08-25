import numpy as np
import tensorflow as tf
from scipy.optimize import nnls
from tensorflow import Tensor

from .Compression import Compression


# from ..utilities.compression_rate import get_sparse_tensor_size_in_bits


class vqSGD(Compression):
    def __init__(self, repetition: int = 1, name="vqSGD"):
        super().__init__(name=name)
        self.s = repetition
        self.compression_rates = []

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
        Quantise gradient: Q(v) = c_i with probability a_i
        c_i of 2*d point set {+- sqrt(d) e_i | i e [d]}
        cp = tf.concat([ d_sqrt * tf.eye(d), - d_sqrt * tf.eye(d)], axis=1)

        probability algorithm:
        for i in range(2 * d):
            if gradient[i % d] > 0 and i <= d - 1:
                a[i] = gradient[i % d] / d_sqrt + gamma / (2 * d)
            elif gradient[i % d] <= 0 and i > d - 1:
                a[i] = -gradient[i % d] / d_sqrt + gamma / (2 * d)
            else:
                a[i] = gamma / (2 * d)
        """
        input_shape = gradient.shape
        gradient = tf.where(tf.math.is_nan(gradient), 0., gradient)

        l2 = tf.norm(gradient, ord=2)
        # if l2 != 0:
        if l2 > 1:
            gradient = tf.reshape(gradient, [-1]) / l2
        else:
            gradient = tf.reshape(gradient, [-1])

        d = gradient.shape[0]
        d_sqrt = np.sqrt(d)

        a = np.zeros(2 * d)

        gamma = 1 - tf.norm(gradient, ord=1) / d_sqrt
        gamma_by_2d = gamma / (2 * d)

        a[:d] = tf.cast(gradient > 0, tf.float32) * ((gradient / d_sqrt) + gamma_by_2d)
        a[d:] = tf.cast(gradient <= 0, tf.float32) * ((-gradient / d_sqrt) + gamma_by_2d)

        a = tf.where(a == 0, gamma_by_2d, a)

        a = a.numpy()
        a[np.isnan(a)] = 0
        np.divide(a, a.sum(), out=a)

        indices = np.random.choice(np.arange(2 * d), self.s, p=a)
        compressed_gradient = np.zeros(d)

        for index in indices:
            if index >= d:
                compressed_gradient[index - d] -= d_sqrt
            else:
                compressed_gradient[index] += d_sqrt

        # print(sum(compressed_gradient))

        compressed_gradient = tf.reshape(compressed_gradient, input_shape) / self.s
        compressed_gradient = tf.cast(compressed_gradient, dtype=variable.dtype)

        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                compressed_gradient)
            self.compression_rates.append(self.cr[variable.ref()])

        return compressed_gradient * l2
