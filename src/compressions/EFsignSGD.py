import numpy as np
import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class EFsignSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, name="EFsignSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)

    def build(self, var_list):
        """Initialize optimizer variables.

        EFsignSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build EFsignSGD variables on.
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

        d = variable.shape.as_list()[0]

        p_t = lr * gradient + self.error[variable.ref()]
        delta_t = (tf.norm(p_t, ord=1)/d) * tf.sign(p_t)

        # update residual error
        self.error[variable.ref()].assign(p_t - delta_t)

        # update iterate
        variable.assign_add(-delta_t)

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




