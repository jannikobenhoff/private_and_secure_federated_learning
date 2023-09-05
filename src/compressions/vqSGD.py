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

                print(np.mean(self.compression_rates), self.s * np.log2(2 * d))

            return compressed_gradient

    def federated_compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int):
        indices_list = []
        l2_scale = []
        for i, gradient in enumerate(gradients):
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
            indices_list.append(indices)
            l2_scale.append(l2)

            if variables[i].ref() not in self.cr:
                self.cr[variables[i].ref()] = flat_gradient.dtype.size * 8 * np.prod(
                    flat_gradient.shape.as_list()) / (self.s * np.log2(2 * d))  # tf.int32.size * 8)  #

                self.compression_rates.append(self.cr[variables[i].ref()])

                print(np.mean(self.compression_rates), self.s * np.log2(2 * d))

        return {
            "compressed_grad": indices_list,
            "decompress_info": l2_scale
        }

    def federated_decompress(self, info, variables):
        decompressed_cradients = []
        for i, indices in enumerate(info["compressed_grad"]):
            l2 = info["decompress_info"][i]
            d = tf.size(variables[i]).numpy()
            d_sqrt = np.sqrt(d)
            decompressed_gradient = np.zeros(d)

            np.add.at(decompressed_gradient, indices[indices < d], d_sqrt)
            np.add.at(decompressed_gradient, indices[indices >= d] - d, -d_sqrt)

            decompressed_gradient = tf.reshape(decompressed_gradient, variables[i].shape) / self.s

            if variables[i].dtype != decompressed_gradient.dtype:
                decompressed_gradient = tf.cast(decompressed_gradient, dtype=variables[i].dtype)

            decompressed_gradient = tf.multiply(decompressed_gradient, l2)
            decompressed_cradients.append(decompressed_gradient)

        return decompressed_cradients
