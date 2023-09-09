import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Tensor

from .Compression import Compression


# from ..utilities.huffman import *


class SparseGradient(Compression):
    def __init__(self, drop_rate: float = 90, name="SparseGradient"):
        super().__init__(name=name)
        self.residuals = None
        self.drop_rate = drop_rate
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        SparseGradient optimizer has one variable:`residuals`.

        Args:
          var_list: list of model variables to build SparseGradient variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.residuals = {}
        flattened_grads = [tf.reshape(var, [-1]) for var in var_list]
        gradient_size = tf.concat(flattened_grads, axis=0)

        for client in range(1, clients + 1):
            self.residuals[str(client)] = tf.Variable(tf.zeros_like(gradient_size, dtype=var_list[0].dtype))
        self._built = True

    def compress(self, grads: list[Tensor], variables, client_id=1):
        """
        Remember residuals (dropped values) locally to add to next gradient
        before dropping again.
        """
        normalized_gradients = []
        for grad, var in zip(grads, variables):
            if grad is not None:
                mean, variance = tf.nn.moments(grad, axes=[0])
                normalized_grad = tf.nn.batch_normalization(
                    grad, mean, variance, offset=None, scale=None, variance_epsilon=1e-5)
                normalized_gradients.append(normalized_grad)
            else:
                normalized_gradients.append(grad)

        flattened_grads = [tf.reshape(grad, [-1]) for grad in normalized_gradients]
        gradient = tf.concat(flattened_grads, axis=0)

        res = self.residuals[str(client_id)]
        gradient_with_residuals = gradient + res

        gradient_dropped = self.gradDrop(gradient_with_residuals, self.drop_rate)
        self.residuals[str(client_id)].assign(gradient - gradient_dropped)

        if variables[0].ref() not in self.cr:
            self.cr[variables[0].ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                gradient_dropped)
            self.compression_rates.append(self.cr[variables[0].ref()])
            self.compression_rates = [np.mean(self.compression_rates)]

        compressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(gradient_dropped[start: start + size])
            compressed_grads.append(tf.reshape(segment, var.shape))
            start += size

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": None,
            "needs_decompress": False
        }

    def gradDrop(self, gradient: Tensor, drop_rate) -> Tensor:
        """
        Updates by removing drop_rate % of the smallest gradients by absolute value
        """
        # threshold = tfp.stats.percentile(tf.abs(gradient), q=drop_rate, interpolation="lower")
        # gradient_dropped = tf.where(tf.abs(gradient) > threshold, gradient, 0)
        # return gradient_dropped

        flattened_tensor: Tensor = tf.reshape(gradient, [-1])
        k = int(np.ceil(flattened_tensor.shape[0] * (1 - drop_rate / 100)))

        gradient_dropped = self.top_k_sparsification(gradient, k)
        return gradient_dropped
