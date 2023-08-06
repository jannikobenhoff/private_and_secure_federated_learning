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

    def build(self, var_list):
        """Initialize optimizer variables.

        SparseGradient optimizer has one variable:`residuals`.

        Args:
          var_list: list of model variables to build SparseGradient variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.residuals = {}
        for var in var_list:
            self.residuals[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="residual"
            )

        #self.compression_rate = var_list[0].dtype.size * 8
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        """
        Remember residuals (dropped values) locally to add to next gradient
        before dropping again.
        """

        res = self.residuals[variable.ref()]
        gradient_with_residuals = gradient + res

        gradient_dropped = self.gradDrop(gradient_with_residuals, self.drop_rate)
        self.residuals[variable.ref()].assign(gradient - gradient_dropped)

        self.compression_rates.append(gradient.dtype.size*8*np.prod(gradient.shape.as_list())/self.get_sparse_tensor_size_in_bits(gradient_dropped))

        return gradient_dropped

    @staticmethod
    def gradDrop(gradient: Tensor, drop_rate) -> Tensor:
        """
        Updates by removing drop_rate % of the smallest gradients by absolute value
        """
        threshold = tfp.stats.percentile(tf.abs(gradient), q=drop_rate, interpolation="lower")
        gradient_dropped = tf.where(tf.abs(gradient) > threshold, gradient, 0)
        return gradient_dropped

    @staticmethod
    def get_sparse_tensor_size_in_bits(tensor):
        num_nonzero_entries = tf.math.count_nonzero(tensor)
        num_index_bits = np.ceil(np.log2(len(tf.reshape(tensor, [-1]))))
        num_value_bits = tensor.dtype.size * 8
        return num_nonzero_entries.numpy() * (num_index_bits + num_value_bits) if num_nonzero_entries.numpy() * (
                num_index_bits + num_value_bits) != 0 else 1
