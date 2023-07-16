import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor

class OneBitSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, name="OneBitSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.quantization_threshold = 0

    def build(self, var_list):
        """Initialize optimizer variables.

        OneBitSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.error = {}
        for var in var_list:
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error"
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        gradient_quantized = tf.sign(gradient + self.error[variable.ref()])
        error = gradient - self.unquantize(gradient_quantized)

        self.error[variable.ref()].assign(error)

        variable.assign_add(-gradient_quantized * lr)

    @staticmethod
    def unquantize(gradients):
        pass

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
