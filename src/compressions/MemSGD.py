import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class MemSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, momentum, name="MemSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables.

        MemSGD optimizer has one variable `momentum`

        Args:
          var_list: list of model variables to build MemSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m", initial_value=0
                )
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]


        m.assign(1)

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




