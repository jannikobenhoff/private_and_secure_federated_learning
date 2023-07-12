import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


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

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        variable.assign_add(-tf.sign(gradient) * lr)

        # Function to calculate p0i
        def p0i(g, P, κ):
            min_vals = [0] * len(g)

            for i in range(len(g)):
                if (g[i] / P[i]) > 0:
                    min_vals[i] = κ * (g[i] / P[i])
                else:
                    min_vals[i] = 1

            return min_vals

        # declaring the inputs
        g = [10, -20, 11]
        P = [15, 9, 10]
        κ = 0.5

        p0 = p0i(g, P, κ)

        j = 0

        # Algorithm 3 Greedy algorithm starts from here
        # repeat
        while (j < P.length - 1):
            # Set
            for i in range(len(g)):
                # Calculate
                pj_i = min(κ * (g[i] / P[i]), 1)

            # Increment
            j = j + 1

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
