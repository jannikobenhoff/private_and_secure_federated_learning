import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression
# from ..utilities.compression_rate import get_sparse_tensor_size_in_bits
# from ..utilities.huffman import *


class TernGrad(Compression):
    def __init__(self, clip: float = 2.5, name="TernGrad"):
        super().__init__(name=name)
        self.clip = clip
        self.compression_rates = []

    def build(self, var_list):
        """Initialize optimizer variables.

        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.compression_rates.append(var_list[0].dtype.size*8/np.log2(3))
        self._built = True

    def compress(self, gradient: Tensor, variable):
        gradient_clip = self.gradient_clipping(gradient, self.clip)
        gradient_tern = self.ternarize(gradient_clip)
        # huffman
        # rle = run_length_encoding(gradient_tern)
        # vc = count_tensor_values(rle)
        # huf = generate_huffman(vc)
        # enc = encode_huffman(rle, huf)
        # print(enc, huf)
        # print((32+len(tf.reshape(gradient_tern, [-1])) * 2) / (len("".join(enc)) + len(huf) * 4+32), 16)
        #
        # return enc, huf, gradient_tern.shape
        return gradient_tern

    @staticmethod
    def ternarize(input_tensor: Tensor) -> Tensor:
        """
        Layer-wise ternarize

        g_t_i_tern = s_t * sign(g_t_i) o b_t
        s_t = max(abs(g_t_i)) = ||g_t_i||∞ (max norm)
        o : Hadamard product
        """
        if len(input_tensor.shape) == 1:
            abs_input = tf.abs(input_tensor)
            s_t = tf.reduce_max(abs_input, axis=0, keepdims=True)
            b_t = tf.cast(abs_input / s_t >= 0.5, input_tensor.dtype)
            return s_t * tf.sign(input_tensor) * b_t
        abs_input = tf.abs(input_tensor)
        s_t = tf.reduce_max(abs_input, axis=1, keepdims=True)
        b_t = tf.cast(abs_input / s_t >= 0.5, dtype=input_tensor.dtype)
        return s_t * tf.sign(input_tensor) * b_t

    @staticmethod
    def gradient_clipping(input_tensor: Tensor, c) -> Tensor:
        """
        Clips the gradient tensor.
        Sigma is the standard deviation of the gradient vector
        c is a constant
        """
        sigma = tf.math.reduce_std(input_tensor)
        abs_input = tf.abs(input_tensor)
        clipped_gradient = tf.where(
            abs_input <= c * sigma, input_tensor, tf.sign(input_tensor) * c * sigma)
        return clipped_gradient
