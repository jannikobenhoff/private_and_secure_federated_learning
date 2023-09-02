import numpy as np
import tensorflow as tf
import torch
from scipy.optimize import nnls
from tensorflow import Tensor

from .Compression import Compression


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
        if True:  # "GPU" not in [d.device_type for d in tf.config.get_visible_devices()]:
            input_shape = gradient.shape
            flat_gradient = tf.reshape(gradient, [-1])

            l2 = tf.norm(flat_gradient, ord=2)
            if l2 != 0:
                flat_gradient = tf.divide(flat_gradient, l2)
            else:
                print("L2 ZERO")

            d = flat_gradient.shape[0]
            d_sqrt = np.sqrt(d)

            gamma = 1 - (tf.norm(flat_gradient, ord=1) / d_sqrt)

            gamma = gamma.numpy()
            gamma_by_2d = gamma / (2 * d)

            a = np.zeros(2 * d)
            a[:d] = (flat_gradient > 0).numpy() * ((flat_gradient.numpy() / d_sqrt) + gamma_by_2d)
            a[d:] = (flat_gradient <= 0).numpy() * ((-flat_gradient.numpy() / d_sqrt) + gamma_by_2d)

            a = np.where(a == 0, gamma_by_2d, a)

            np.divide(a, a.sum(), out=a)

            indices = np.random.choice(np.arange(2 * d), self.s, p=a)

            compressed_gradient = np.zeros(d)

            np.add.at(compressed_gradient, indices[indices < d], d_sqrt)
            np.add.at(compressed_gradient, indices[indices >= d] - d, -d_sqrt)

            compressed_gradient = tf.reshape(compressed_gradient, input_shape) / self.s

            if flat_gradient.dtype != compressed_gradient.dtype:
                compressed_gradient = tf.cast(compressed_gradient, dtype=flat_gradient.dtype)

            compressed_gradient = tf.multiply(compressed_gradient, l2)

            if variable.ref() not in self.cr:
                # self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                #     gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                #     compressed_gradient)

                self.cr[variable.ref()] = flat_gradient.dtype.size * 8 * np.prod(
                    flat_gradient.shape.as_list()) / min((self.s * np.log2(2 * d)), self.get_sparse_tensor_size_in_bits(
                    compressed_gradient))  # tf.int32.size * 8)  #

                self.compression_rates.append(self.cr[variable.ref()])
                print(tf.unique_with_counts(
                    tf.sign(tf.reshape(compressed_gradient, [-1])) * tf.sign(tf.reshape(gradient, [-1])))[0])
                print(tf.unique_with_counts(
                    tf.sign(tf.reshape(compressed_gradient, [-1])) * tf.sign(tf.reshape(gradient, [-1])))[2])
                print(np.mean(self.compression_rates))

            return compressed_gradient
        else:

            input_shape = gradient.shape
            flat_gradient = tf.reshape(gradient, [-1])

            l2 = tf.norm(flat_gradient, ord=2)
            flat_gradient = tf.where(l2 != 0, flat_gradient / l2, 0.0)
            # tf.where will handle the condition for each element of the tensor.

            d = flat_gradient.shape[0]
            d_sqrt = tf.sqrt(tf.cast(d, tf.float32))

            gamma = 1 - (tf.norm(flat_gradient, ord=1) / d_sqrt)
            gamma_by_2d = gamma / (2 * d)

            a_positive = tf.where(flat_gradient > 0, flat_gradient / d_sqrt + gamma_by_2d, gamma_by_2d)
            a_negative = tf.where(flat_gradient <= 0, -flat_gradient / d_sqrt + gamma_by_2d, gamma_by_2d)

            a = tf.concat([a_positive, a_negative], 0)
            a /= tf.reduce_sum(a)

            indices = tf.random.categorical(tf.math.log([a]), self.s)

            with tf.device('/CPU:0'):
                indices = indices[0].numpy()

                compressed_gradient = np.zeros(d)

                np.add.at(compressed_gradient, indices[indices < d], d_sqrt)
                np.add.at(compressed_gradient, indices[indices >= d] - d, -d_sqrt)

                compressed_gradient = tf.convert_to_tensor(compressed_gradient)

            compressed_gradient = tf.reshape(compressed_gradient, input_shape) / self.s

            if flat_gradient.dtype != compressed_gradient.dtype:
                compressed_gradient = tf.cast(compressed_gradient, dtype=flat_gradient.dtype)

            compressed_gradient *= l2

            if variable.ref() not in self.cr:
                self.cr[variable.ref()] = flat_gradient.dtype.size * 8 * np.prod(
                    flat_gradient.shape.as_list()) / min((self.s * np.log2(2 * d)), self.get_sparse_tensor_size_in_bits(
                    compressed_gradient))  # tf.int32.size * 8)  #

                self.compression_rates.append(self.cr[variable.ref()])
                print(np.mean(self.compression_rates))

            return compressed_gradient
