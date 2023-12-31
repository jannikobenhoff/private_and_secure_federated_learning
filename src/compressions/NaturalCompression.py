import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression


class NaturalCompression(Compression):
    def __init__(self, clip_max: int = 10, clip_min: int = -50, name="NaturalCompression"):
        super().__init__(name=name)
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.
        NaturalCompression optimizer has no additional variables.

        Args:
          var_list: list of model variables to build NaturalCompression variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.compression_rates.append(var_list[0].dtype.size)
        self._built = True

    # def compress(self, gradient: Tensor, variable) -> Tensor:
    #     gradient_compressed = self.natural_compress(gradient)
    #     return gradient_compressed

    def compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int = 1):
        compressed_grads = []

        for i, gradient in enumerate(gradients):
            gradient_quantized = self.natural_compress(gradient)
            compressed_grads.append(gradient_quantized)

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": None,
            "needs_decompress": False
        }

    def natural_compress(self, input_array):
        """
        Performing a randomized logarithmic rounding of input array
        """
        abs_array = np.abs(input_array)
        a = np.log(abs_array) / np.log(2.0)

        a_up = np.ceil(a)
        a_up = np.clip(a_up, a_min=self.clip_min, a_max=self.clip_max)

        a_down = np.floor(a)
        a_down = np.clip(a_down, a_min=self.clip_min, a_max=self.clip_max)

        p_down = (np.power(2.0, a_up) - abs_array) / (np.power(2.0, a_up) - np.power(2.0, a_down))
        rand = np.random.uniform(size=p_down.shape)

        return np.sign(input_array) * np.where(p_down > rand, np.power(2.0, a_down), np.power(2.0, a_up))

    # def natural_compress(self, input_tensor: Tensor) -> Tensor:
    #     """
    #     Performing a randomized logarithmic rounding of input t
    #     """
    #     abs_tensor = tf.abs(input_tensor)
    #     a = tf.experimental.numpy.log2(abs_tensor)
    #     a_up = tf.math.ceil(a)
    #     tf.clip_by_value(a_up, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
    #     a_down = tf.math.floor(a)
    #     tf.clip_by_value(a_down, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
    #
    #     p_down = (tf.pow(2.0, a_up) - abs_tensor) / tf.pow(2.0, a_down)
    #     rand = tf.random.uniform(shape=p_down.shape)
    #
    #     return tf.sign(input_tensor) * tf.where(p_down > rand, tf.pow(2.0, a_down), tf.pow(2.0, a_up))

    # def natural_compress(self, input_tensor: tf.Tensor) -> tf.Tensor:
    #     """
    #     Performing a randomized logarithmic rounding of input tensor
    #     """
    #     abs_tensor = tf.abs(input_tensor)
    #     a = tf.math.log(abs_tensor) / tf.math.log(2.0)
    #
    #     a_up = tf.math.ceil(a)
    #     a_up = tf.clip_by_value(a_up, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
    #
    #     a_down = tf.math.floor(a)
    #     a_down = tf.clip_by_value(a_down, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
    #
    #     p_down = (tf.pow(2.0, a_up) - abs_tensor) / (tf.pow(2.0, a_up) - tf.pow(2.0, a_down))
    #     rand = tf.random.uniform(shape=p_down.shape, dtype=p_down.dtype)
    #
    #     return tf.sign(input_tensor) * tf.where(p_down > rand, tf.pow(2.0, a_down), tf.pow(2.0, a_up))
