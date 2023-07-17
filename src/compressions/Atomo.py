import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor
import tensorflow_probability as tfp


class Atomo(optimizer.Optimizer):
    def __init__(self, learning_rate, sparsity_budget: int, name="Atomo"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.sparsity_budget = sparsity_budget

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
        gradient_estimator = self.atomic_sparsification(gradient, s=self.sparsity_budget)

        variable.assign_add(-gradient_estimator*lr)

    @staticmethod
    def atomic_sparsification(gradient: Tensor, s: int) -> Tensor:
        """
        Basic implementation of entry-wise decomposition: lambda_i = gradient_i
        """
        i = 0
        n = gradient.shape[0]
        p = tf.ones_like(gradient, dtype=gradient.dtype)

        while i <= n:
            if False:
                i = n + 1
            else:
                i += 1
                s -= 1

        t = tfp.distributions.Bernoulli(probs=p, dtype=gradient.dtype).sample()

        gradient_estimator = gradient * t / p
        return gradient_estimator

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
