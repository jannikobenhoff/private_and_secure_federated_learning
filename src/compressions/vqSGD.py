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

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        vqSGD optimizer has no variables.

        Args:
          var_list: list of model variables to build vqSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, grads: list[Tensor], variables):
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
        flattened_grads = [tf.reshape(grad, [-1]) for grad in grads]
        gradient = tf.concat(flattened_grads, axis=0)

        l2 = tf.norm(gradient, ord=2)
        if l2 != 0:
            gradient = tf.divide(gradient, l2)
        else:
            print("L2 ZERO")

        d = gradient.shape[0]
        d_sqrt = np.sqrt(d)

        gamma = 1 - (tf.norm(gradient, ord=1) / d_sqrt)

        gamma = gamma.numpy()
        gamma_by_2d = gamma / (2 * d)

        a = np.zeros(2 * d)
        a[:d] = (gradient > 0).numpy() * ((gradient.numpy() / d_sqrt) + gamma_by_2d)
        a[d:] = (gradient <= 0).numpy() * ((-gradient.numpy() / d_sqrt) + gamma_by_2d)

        a = np.where(a == 0, gamma_by_2d, a)

        np.divide(a, a.sum(), out=a)

        indices = np.random.choice(np.arange(2 * d), self.s, p=a)

        if variables[0].ref() not in self.cr:
            self.cr[variables[0].ref()] = d / (self.s * np.log2(2 * d))
            self.compression_rates.append(self.cr[variables[0].ref()])
            print("CR:", np.mean(self.compression_rates))

        return {
            "compressed_grads": indices,
            "decompress_info": l2,
            "needs_decompress": True
        }

    def decompress(self, compressed_data, variables):
        indices = compressed_data["compressed_grads"]
        l2 = compressed_data["decompress_info"]
        d = sum([tf.size(var).numpy() for var in variables])
        d_sqrt = np.sqrt(d)

        decompressed_grad = np.zeros(d)

        np.add.at(decompressed_grad, indices[indices < d], d_sqrt)
        np.add.at(decompressed_grad, indices[indices >= d] - d, -d_sqrt)

        decompressed_grad = decompressed_grad / self.s

        decompressed_grad = tf.multiply(decompressed_grad, l2)

        decompressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(decompressed_grad[start: start + size])
            decompressed_grads.append(tf.reshape(segment, var.shape))
            start += size

        return decompressed_grads
