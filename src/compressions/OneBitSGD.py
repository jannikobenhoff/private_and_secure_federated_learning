import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from src.compressions.Compression import Compression
from src.utilities.compression_rate import get_compression_rate


class OneBitSGD(Compression):
    def __init__(self, name="OneBitSGD"):
        super().__init__(name=name)
        self.quantization_threshold = 0

    def build(self, var_list):
        """Initialize optimizer variables.

        OneBitSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.error = {}
        for var in var_list:
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error"
            )
        self._built = True

    def compress(self, gradient: Tensor, variable):
        gradient_quantized = tf.sign(gradient + self.error[variable.ref()])
        error = gradient - self.unquantize(gradient_quantized, gradient)

        self.error[variable.ref()].assign(error)
        # get_compression_rate(gradient, gradient_quantized)
        # gradient_quantized = tf.cast(gradient_quantized, dtype=tf.int8)
        return gradient_quantized

    @staticmethod
    def unquantize(gradient_quantized: Tensor, gradient: Tensor):
        x = tf.reshape(gradient_quantized, [-1]).numpy()
        y = tf.reshape(gradient, [-1]).numpy()
        indices_minus1 = np.where(x == -1)
        indices_1 = np.where(x == 1)

        minus1_values = y[indices_minus1]
        values_1 = y[indices_1]

        a = np.mean(minus1_values)
        b = np.mean(values_1)

        return tf.where(gradient_quantized == -1, a, b)
