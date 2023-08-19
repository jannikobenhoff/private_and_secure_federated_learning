import numpy as np
import tensorflow as tf
from tensorflow import Tensor

# from ..utilities.compression_rate import get_sparse_tensor_size_in_bits
from .Compression import Compression


class GradientSparsification(Compression):
    def __init__(self, k: float = 0.02, max_iter: int = 2, name="GradientSparsification"):
        super().__init__(name=name)
        self.k = float(k)
        self.max_iter = int(max_iter)
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
        gradient = tf.where(tf.math.is_nan(gradient), 0., gradient)

        probabilities = self.greedy_algorithm(input_tensor=gradient, max_iter=self.max_iter, k=self.k)

        rand = tf.random.uniform(shape=probabilities.shape, minval=0, maxval=1)
        selectors = tf.where(probabilities > rand, 1.0, 0.0)

        gradient_spars = tf.multiply(selectors, gradient) / probabilities

        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                gradient_spars)
            self.compression_rates.append(self.cr[variable.ref()])

        gradient_spars = tf.reshape(gradient_spars, input_shape)
        gradient_spars = tf.where(tf.math.is_nan(gradient_spars), 0., gradient_spars)

        return gradient_spars

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, k: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        p = tf.ones_like(input_tensor)
        d = tf.cast(tf.shape(input_tensor)[0], dtype=tf.float32)
        k_d = k * d

        comp = k * d * tf.abs(input_tensor) / tf.reduce_sum(tf.abs(input_tensor)).numpy()
        comp = tf.where(tf.math.is_nan(comp), 0., comp)
        p = tf.where(comp < 1,
                     comp, p)
        c = 2
        j = 0

        while j < max_iter and c > 1:
            active_set = tf.where(p != 1, p, 0)
            cardinality = tf.cast(tf.math.count_nonzero(active_set), dtype=tf.float32)
            sum_active_set = tf.reduce_sum(active_set)

            c = (k_d - d + cardinality) / sum_active_set
            p = tf.minimum(c * p, 1)

            j += 1

        return p
