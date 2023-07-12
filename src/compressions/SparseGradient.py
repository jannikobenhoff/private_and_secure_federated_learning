import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
import tensorflow_probability as tfp
from tensorflow import Tensor


class SparseGradient(optimizer.Optimizer):
    def __init__(self, learning_rate, drop_rate, name="SparseGradient"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.drop_rate = drop_rate

    def build(self, var_list):
        """Initialize optimizer variables.

        SparseGradient optimizer has two variables: `drop_rate` and `residuals`.

        Args:
          var_list: list of model variables to build SparseGradient variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.residuals = []
        for var in var_list:
            self.residuals.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="residual"
                )
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        """
        Remember residuals (dropped values) locally to add to next gradient
        before dropping again.
        """
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        var_key = self._var_key(variable)
        res = self.residuals[self._index_dict[var_key]]

        gradient_with_residuals = gradient + res

        gradient_dropped = self.gradDrop(gradient_with_residuals, self.drop_rate)
        self.residuals[self._index_dict[var_key]].assign(gradient - gradient_dropped)

        variable.assign_add(- gradient_dropped * lr)

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
    def gradDrop(gradient: Tensor, drop_rate) -> Tensor:
        """
        Updates by removing drop_rate % of the smallest gradients by absolute value
        """
        threshold = tfp.stats.percentile(tf.abs(gradient), drop_rate)
        gradient_dropped = tf.where(tf.abs(gradient) >= threshold, gradient, 0)
        return gradient_dropped

