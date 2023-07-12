import numpy as np
import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class Atomo(optimizer.Optimizer):
    def __init__(self, learning_rate, momentum, name="Atomo"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables.

        Atomo optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build Atomo variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.error = {}
        for var in var_list:
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error",
                initial_value=tf.zeros(shape=tf.shape(var))
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        # update residual error
        self.error[variable.ref()].assign()

        # update iterate
        variable.assign_add()

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
