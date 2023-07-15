import numpy as np
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
        probabilities = self.greedy_algorithm(input_tensor=gradient, max_iter=2, k=0.001)
        selectors = np.random.randint(2, size=probabilities.shape)
        gradient_spars = selectors * gradient / probabilities
        variable.assign_add(- gradient_spars * lr)

    @staticmethod
    def greedy_algorithm(input_tensor: Tensor, k: float, max_iter: int) -> Tensor:
        """
        Finds the optimal probability vector p in max_iter iterations.

        Returns:
          The optimal probability vector p.
        """
        j = 0
        p = tf.ones_like(input_tensor)
        d = input_tensor.shape[0]

        comp = k*d*tf.abs(input_tensor)/tf.reduce_sum(input_tensor).numpy()
        p = tf.where(comp < 1,
                     comp, p)
        c = 2
        while j < max_iter and c > 1:
            active_set = tf.where(p != 1, p, 0)
            cardinality = tf.reduce_sum(tf.where(active_set != 0, 1, 0)).numpy()
            c = (k*d-d+cardinality)/tf.reduce_sum(active_set).numpy()
            cp = tf.math.multiply(c, p)
            p = tf.where(cp < 1,
                         cp, 1)
            j += 1
        return p

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
