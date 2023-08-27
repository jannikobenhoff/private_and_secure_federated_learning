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

        # rand = tf.random.uniform(shape=probabilities.shape, minval=0, maxval=1)
        # selectors = tf.where(probabilities > rand, 1.0, 0.0)
        random_tensor = tf.random.uniform(probabilities.shape, 0, 1)
        selectors = tf.cast(tf.less(random_tensor, probabilities), gradient.dtype)

        gradient_spars = tf.multiply(selectors, gradient) / probabilities
        # gradient_spars = tf.where(probabilities > 0, gradient_spars / probabilities, 0)
        gradient_spars = tf.where(tf.math.is_nan(gradient_spars), 0., gradient_spars)

        if True:  # variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                gradient_spars)
            self.compression_rates.append(self.cr[variable.ref()])

        gradient_spars = tf.reshape(gradient_spars, input_shape)

        return gradient_spars

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, k: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        p = tf.ones_like(input_tensor)
        d = tf.cast(tf.shape(input_tensor)[0], dtype=input_tensor.dtype)
        k_d = tf.multiply(k, d)

        comp = k_d * tf.abs(input_tensor) / tf.reduce_sum(tf.abs(input_tensor))
        # comp = tf.where(tf.math.is_nan(comp), 0., comp)
        p = tf.where(comp < 1,
                     comp, p)
        c = tf.constant(2, dtype=input_tensor.dtype)
        j = tf.constant(0, dtype=input_tensor.dtype)

        while j < max_iter and c > 1:
            # print("ITER")
            active_set = tf.where(p != 1, 1.0, 0)
            cardinality = tf.cast(tf.math.count_nonzero(active_set), dtype=input_tensor.dtype)
            sum_active_set = tf.reduce_sum(active_set)

            c = (k_d - d + cardinality) / sum_active_set
            p = tf.minimum(c * p, 1)
            # print(c)
            j = tf.add(j, 1)
        return p
