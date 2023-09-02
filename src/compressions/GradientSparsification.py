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
        """
        Q(g) = [Z1 * g1/p1, ...]
        Z1: selector
        g1: gradient
        p1: probability calculated with greedy alg.
        """
        input_shape = gradient.shape
        gradient = tf.reshape(gradient, [-1])

        probabilities = self.greedy_algorithm(input_tensor=gradient, max_iter=self.max_iter, kappa=self.k)

        random_tensor = tf.random.uniform(probabilities.shape, 0, 1)
        selectors = tf.cast(tf.less(random_tensor, probabilities), gradient.dtype)

        gradient_spars = tf.multiply(selectors, gradient) / probabilities
        gradient_spars = tf.where(tf.math.is_nan(gradient_spars), 0., gradient_spars)

        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                gradient_spars)
            self.compression_rates.append(self.cr[variable.ref()])
            self.compression_rates = [np.mean(self.compression_rates)]

            # print(np.mean(self.compression_rates))

        gradient_spars = tf.reshape(gradient_spars, input_shape)

        return gradient_spars

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, kappa: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        g = input_tensor.numpy()
        d = len(g)
        p = np.minimum(kappa * d * np.abs(g) / np.sum(np.abs(g)), 1)
        j = 0

        while j < max_iter:
            # Identify active set I
            active_set = np.where(p != 1)[0]

            # If active set is empty, break
            if len(active_set) == 0:
                break

            c = (kappa * d - d + len(active_set)) / np.sum(p[active_set])

            p = np.minimum(c * p, 1)

            j += 1

            # Check termination condition
            if c <= 1:
                break
        return p
