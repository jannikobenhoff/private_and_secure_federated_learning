import tensorflow as tf
from keras.optimizers.optimizer import Optimizer
from tensorflow import Tensor


class EFsignSGD(Optimizer):
    def __init__(self, learning_rate, name="EFsignSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.compression_rates = []
    def build(self, var_list):
        """Initialize optimizer variables.

        EFsignSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build EFsignSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.errors = []
        for var in var_list:
            self.errors.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="error"
                )
            )
        self.compression_rates.append(var_list[0].dtype.size*8)
        self._built = True

    def update_step(self, gradient: Tensor, variable) -> Tensor:
        """
        Send sign(p_t) and norm(p_t)
        -> compression rate like signSGD + bits for norm
        """
        var_key = self._var_key(variable)

        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        error = self.errors[self._index_dict[var_key]]

        d = variable.shape.as_list()[0]
        p_t = lr * gradient + error

        norm = tf.norm(p_t, ord=1, axis=0, keepdims=True)/d
        # TODO nan check
        # delta_t = (tf.norm(p_t, ord=1, keepdims=True, axis=1)/d) * tf.sign(p_t)
        delta_t = tf.multiply(norm, tf.sign(p_t))

        # update residual error
        error.assign(p_t-delta_t)
        # update iterate
        # variable.assign_add(-delta_t)

        return delta_t

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
