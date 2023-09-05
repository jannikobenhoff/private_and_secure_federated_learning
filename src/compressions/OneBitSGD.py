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
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.error = {}
        if clients == 1:
            # Local Setup
            for var in var_list:
                self.error[var.ref()] = self.add_variable_from_reference(
                    model_variable=var, variable_name="error", initial_value=tf.zeros_like(var)
                )
        else:
            # Federated Setup
            for client_id in range(1, clients + 1):
                for var in var_list:
                    self.error[var.name + str(client_id)] = self.add_variable_from_reference(
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

    def federated_compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int):
        quantized_gradients = []
        decomp = []
        for i, gradient in enumerate(gradients):
            quantized_grad = tf.sign(gradient + self.error[variables[i].name + str(client_id)])
            quantized_gradients.append(quantized_grad)

            (a, b) = self.unquantize_info(quantized_grad, gradient)
            self.error[variables[i].name + str(client_id)].assign(gradient - tf.where(quantized_grad == -1, a, b))
            decomp.append((a, b))
        return {
            'compressed_grad': quantized_gradients,
            'decompress_info': decomp
        }

    def unquantize_info(self, quantized_grad, gradient):
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

    def federated_decompress(self, info, variables):
        decompressed_gradients = []
        for i, gradient in enumerate(info["compressed_grad"]):
            (a, b) = info["decompress_info"][i]
            decompressed_gradients.append(tf.where(gradient == -1, a, b))

        return decompressed_gradients
