import tensorflow as tf
from tensorflow import Tensor
from keras.optimizers.optimizer_experimental import optimizer


class NaturalCompression(optimizer.Optimizer):
    def __init__(self, learning_rate, name="NaturalCompression"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)

    def build(self, var_list):
        """Initialize optimizer variables.

        NaturalCompression optimizer has no variable.

        Args:
          var_list: list of model variables to build NaturalCompression variables on.
        """
        return

    def _update_step(self, gradient, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        gradient_compressed = self.compress(gradient)
        variable.assign_add(-gradient_compressed * lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                )
            }
        )
        return config

    @staticmethod
    def compress(input_tensor: Tensor) -> Tensor:
        """
        Performing a randomized logarithmic rounding of input t
        """
        abs_tensor = tf.abs(input_tensor)
        a = tf.experimental.numpy.log2(abs_tensor)
        a_up = tf.math.ceil(a)
        a_down = tf.math.floor(a)
        p_down = (tf.pow(2.0, a_up) - abs_tensor) / tf.pow(2.0, a_down)
        return tf.sign(input_tensor) * tf.where(p_down >= 0.5, tf.pow(2.0, a_down), tf.pow(2.0, a_up))
