import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression


class TopK(Compression):
    def __init__(self, k, name="TopK"):
        super().__init__(name=name)
        self.k = k
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, grads: list[Tensor], variables):
        flattened_grads = [tf.reshape(grad, [-1]) for grad in grads]
        gradient = tf.concat(flattened_grads, axis=0)

        sparse_gradient = self.top_k_sparsification(gradient, self.k)

        if variables[0].ref() not in self.cr:
            self.cr[variables[0].ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                sparse_gradient)
            self.compression_rates.append(self.cr[variables[0].ref()])
            self.compression_rates = [np.mean(self.compression_rates)]

        compressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(sparse_gradient[start: start + size])
            compressed_grads.append(tf.reshape(segment, var.shape))
            start += size

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": None,
            "needs_decompress": False
        }
