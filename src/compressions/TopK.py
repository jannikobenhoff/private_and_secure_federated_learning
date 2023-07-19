import tensorflow as tf
from tensorflow import Tensor

from src.compressions.Compression import Compression


class TopK(Compression):
    def __init__(self, k, name="TopK"):
        super().__init__(name=name)
        self.k = k

    def build(self, var_list):
        """Initialize optimizer variables.

        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        sparse_gradient = self.top_k_sparsification(gradient, self.k)
        return sparse_gradient

    @staticmethod
    def top_k_sparsification(input_tensor: Tensor, k: int) -> Tensor:
        """
        Returns a sparse tensor of the input tensor, with the top k elements.
        k = 2: [1, 2, 3] -> [0, 2, 3]
        """
        input_shape = input_tensor.shape
        flattened_tensor: Tensor = tf.reshape(input_tensor, [-1])

        if tf.size(flattened_tensor) < k:
            return input_tensor

        abs_tensor = tf.abs(flattened_tensor)
        indices = tf.math.top_k(abs_tensor, k).indices

        mask = tf.zeros(flattened_tensor.shape)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1),
                                           tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = tf.math.multiply(flattened_tensor, mask)
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor
