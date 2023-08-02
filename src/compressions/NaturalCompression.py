import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression


class NaturalCompression(Compression):
    def __init__(self, clip_max: int = 10, clip_min: int = -50, name="NaturalCompression"):
        super().__init__(name=name)
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.compression_rate = None

    def build(self, var_list):
        """Initialize optimizer variables.
        NaturalCompression optimizer has no additional variables.

        Args:
          var_list: list of model variables to build NaturalCompression variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.compression_rate = var_list[0].dtype.size

        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        gradient_compressed = self.natural_compress(gradient)
        return gradient_compressed

    def natural_compress(self, input_tensor: Tensor) -> Tensor:
        """
        Performing a randomized logarithmic rounding of input t
        """
        abs_tensor = tf.abs(input_tensor)
        a = tf.experimental.numpy.log2(abs_tensor)
        a_up = tf.math.ceil(a)
        tf.clip_by_value(a_up, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
        a_down = tf.math.floor(a)
        tf.clip_by_value(a_down, clip_value_min=self.clip_min, clip_value_max=self.clip_max)

        p_down = (tf.pow(2.0, a_up) - abs_tensor) / tf.pow(2.0, a_down)
        rand = tf.random.uniform(shape=p_down.shape)

        return tf.sign(input_tensor) * tf.where(p_down > rand, tf.pow(2.0, a_down), tf.pow(2.0, a_up))
