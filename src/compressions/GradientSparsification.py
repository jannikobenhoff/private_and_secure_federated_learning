import numpy as np
import tensorflow as tf
from tensorflow import Tensor

# from ..utilities.compression_rate import get_sparse_tensor_size_in_bits
from .Compression import Compression


class GradientSparsification(Compression):
    def __init__(self, k: float = 0.02, max_iter: int = 2, name="GradientSparsification"):
        super().__init__(name=name)
        self.k = k
        self.max_iter = max_iter
        self.compression_rates = []

    def build(self, var_list):
        """Initialize optimizer variables.
        GradientSparsification optimizer has no variables.

        Args:
          var_list: list of model variables to build GradientSparsification variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        input_shape = gradient.shape
        gradient = tf.reshape(gradient, [-1])

        probabilities = self.greedy_algorithm(input_tensor=gradient, max_iter=self.max_iter, k=self.k)

        rand = tf.random.uniform(shape=probabilities.shape, minval=0, maxval=1)
        selectors = tf.where(probabilities > rand, 1.0, 0.0)

        gradient_spars = tf.multiply(selectors, gradient) / probabilities

        self.compression_rates.append(gradient.dtype.size * 8 * np.prod(gradient.shape.as_list()) /
                                      get_sparse_tensor_size_in_bits(gradient_spars))

        gradient_spars = tf.reshape(gradient_spars, input_shape)
        return gradient_spars

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, k: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        j = 0
        p = tf.ones_like(input_tensor)
        d = input_tensor.shape[0]

        comp = k * d * tf.abs(input_tensor) / tf.reduce_sum(tf.abs(input_tensor)).numpy()
        p = tf.where(comp < 1,
                     comp, p)

        c = 2
        while j < max_iter and c > 1:
            active_set = tf.where(p != 1, p, 0)
            cardinality = tf.math.count_nonzero(active_set).numpy()
            c = (k * d - d + cardinality) / tf.reduce_sum(active_set).numpy()
            cp = tf.math.multiply(c, p)
            p = tf.where(cp < 1, cp, 1)
            j += 1
        return p
