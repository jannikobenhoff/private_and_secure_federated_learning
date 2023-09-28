import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from .Compression import Compression


class OneBitSGD(Compression):
    def __init__(self, name="OneBitSGD"):
        super().__init__(name=name)
        self.quantization_threshold = 0
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        OneBitSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build OneBitSGD variables on.#
          clients: Number of clients. 1 for local training.
        """
        if hasattr(self, "_built") and self._built:
            return
        print("1-Bit SGD built.")

        self.error = {}
        for client_id in range(1, clients + 1):
            for var in var_list:
                self.error[var.name + str(client_id)] = tf.Variable(tf.zeros_like(var), trainable=False)

        self.compression_rates.append(var_list[0].dtype.size * 8)
        """
        1-bit Tensor is sent, as well as values a and b of the un-quantized Tensor.
        """
        self._built = True

    def compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int = 1):
        compressed_grads = []
        decompress_info = []
        for i, gradient in enumerate(gradients):
            gradient_corrected = gradient + self.error[variables[i].name + str(client_id)]
            gradient_quantized = tf.where(gradient_corrected >= 0, 1, -1)
            compressed_grads.append(gradient_quantized)
            (a, b) = self.un_quantize_info(gradient_quantized, gradient)
            self.error[variables[i].name + str(client_id)].assign(gradient - tf.where(gradient_quantized == -1, a, b))
            decompress_info.append((a, b))

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": decompress_info,
            "needs_decompress": True
        }

    def decompress(self, compressed_data, variables):
        decompressed_grads = []
        for i, gradient in enumerate(compressed_data["compressed_grads"]):
            (a, b) = compressed_data["decompress_info"][i]
            decompressed_grads.append(tf.where(gradient == -1, a, b))
        return decompressed_grads

    @staticmethod
    def un_quantize_info(quantized_grad, gradient):
        """
                Reconstruction values to un-quantize the quantized tensor.
                The two values are recomputed as to minimize the square quantization error
                and transmitted in each data exchange.
                """
        x = tf.reshape(quantized_grad, [-1]).numpy()
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

        return a, b
