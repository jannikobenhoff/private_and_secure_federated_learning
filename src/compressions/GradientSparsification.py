import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer


class GradientSparsification(optimizer.Optimizer):
    def __init__(self, learning_rate, name="GradientSparsification"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)

    def build(self, var_list):
        """Initialize optimizer variables.

        GradientSparsification optimizer has one variable ``

        Args:
          var_list: list of model variables to build GradientSparsification variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.error = []
        for var in var_list:
            self.error.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="error"
                )
            )
        self._built = True

    def _update_step(self, gradient, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        variable.assign_add(-tf.sign(gradient) * lr)

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
