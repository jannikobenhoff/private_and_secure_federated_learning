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

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.
        GradientSparsification optimizer has no variables.

        Args:
          var_list: list of model variables to build GradientSparsification variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, grads: list[Tensor], variables, log=True):
        """
        Q(g) = [Z1 * g1/p1, ...]
        Z1: selector
        g1: gradient
        p1: probability calculated with greedy alg.
        """
        flattened_grads = [tf.reshape(grad, [-1]) for grad in grads]
        gradient = tf.concat(flattened_grads, axis=0)

        probabilities = self.greedy_algorithm(input_tensor=gradient, max_iter=self.max_iter, kappa=self.k)

        random_tensor = tf.random.uniform(probabilities.shape, 0, 1)
        selectors = tf.cast(tf.less(random_tensor, probabilities), gradient.dtype)

        gradient_spars = tf.multiply(selectors, gradient) / probabilities
        gradient_spars = tf.where(tf.math.is_nan(gradient_spars), 0., gradient_spars)

        if log:  # variables[0].ref() not in self.cr:
            # self.cr[variables[0].ref()] =
            self.compression_rates.append(gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                gradient_spars))
            self.compression_rates = [np.mean(self.compression_rates)]
            # self.compression_rates = [gradient.dtype.size * 8 * np.prod(
            #     gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
            #     gradient_spars)]

        compressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(gradient_spars[start: start + size])
            compressed_grads.append(tf.reshape(segment, var.shape))
            start += size

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": None,
            "needs_decompress": False
        }

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, kappa: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        g = input_tensor.numpy()
        d = len(g)
        if np.sum(np.abs(g)) == 0:
            return input_tensor
        else:
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
