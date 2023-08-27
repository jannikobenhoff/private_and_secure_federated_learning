import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from .Compression import Compression


class OneBitSGD(Compression):
    def __init__(self, name="OneBitSGD"):
        super().__init__(name=name)
        self.quantization_threshold = 0
        self.compression_rates = []

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
                model_variable=var, variable_name="error", initial_value=tf.zeros_like(var)
            )
        self.compression_rates.append(var_list[0].dtype.size * 8)
        """
        1-bit Tensor is sent, as well as values a and b of the un-quantized Tensor.
        """
        self._built = True

    def compress(self, gradient: Tensor, variable):
        gradient_quantized = tf.sign(gradient + self.error[variable.ref()])

        gradient_un_quantized = self.un_quantize(gradient_quantized, gradient)
        error = gradient - gradient_un_quantized

        self.error[variable.ref()].assign(error)

        return gradient_un_quantized

    @staticmethod
    def un_quantize(gradient_quantized: Tensor, gradient: Tensor):
        """
        Reconstruction values to un-quantize the quantized tensor.
        The two values are recomputed as to minimize the square quantization error
        and transmitted in each data exchange.
        """
        x = tf.reshape(gradient_quantized, [-1]).numpy()
        y = tf.reshape(gradient, [-1]).numpy()
        indices_minus1 = np.where(x == -1.0)
        indices_1 = np.where(x == 1.0)

        minus1_values = y[indices_minus1]
        values_1 = y[indices_1]

        if len(minus1_values) < 1:
            a = 0.0
        else:
            a = np.nanmean(minus1_values)
        if len(values_1) < 1:
            b = 0.0
        else:
            b = np.nanmean(values_1)

        return tf.where(gradient_quantized == -1, a, b)
