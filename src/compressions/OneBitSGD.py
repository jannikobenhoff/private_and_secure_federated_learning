import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from .Compression import Compression
# from ..utilities.huffman import *


class OneBitSGD(Compression):
    def __init__(self, name="OneBitSGD"):
        super().__init__(name=name)
        self.quantization_threshold = 0
        self.compression_rate = None

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
        self.compression_rate = var_list[0].dtype.size*8
        self._built = True

    def compress(self, gradient: Tensor, variable):
        gradient_quantized = tf.sign(gradient + self.error[variable.ref()])
        error = gradient - self.unquantize(gradient_quantized, gradient)

        self.error[variable.ref()].assign(error)
        # get_compression_rate(gradient, gradient_quantized)
        # gradient_quantized = tf.cast(gradient_quantized, dtype=tf.int8)
        # huffman
        # rle = run_length_encoding(gradient_quantized)
        # vc = count_tensor_values(rle)
        # huf = generate_huffman(vc)
        # enc = encode_huffman(rle, huf)
        # print(enc, huf)
        # print((len(tf.reshape(gradient_quantized, [-1])) *32) / (len("".join(enc))))# + len(huf) * 4))
        #
        # return enc, huf, gradient_quantized.shape
        return gradient_quantized

    @staticmethod
    def unquantize(gradient_quantized: Tensor, gradient: Tensor):
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
