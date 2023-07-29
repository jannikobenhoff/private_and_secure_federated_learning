import numpy as np
import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor

from src.utilities.csvec import CSVec


class FetchSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, r: int, c: int, momentum: float = 0.9, name="FetchSGD"):
        super().__init__(name=name)
        self.error = None
        self.momentums = None
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.r = r
        self.c = c


    def build(self, var_list):
        """Initialize optimizer variables.

        FetchSGD optimizer has variables `momentum`, `error`

        Args:
          var_list: list of model variables to build FetchSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        self.error = {}
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m", shape=(self.r, self.c)
                )
            )
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error", shape=(self.r, self.c)
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        input_shape = gradient.shape

        d = tf.reshape(gradient, [-1]).shape[0]
        cs = CSVec(d, self.c, self.r)

        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        momentum = tf.cast(self.momentum, variable.dtype.base_dtype)
        var_key = self._var_key(variable)
        m_old = self.momentums[self._index_dict[var_key]]
        error = self.error[variable.ref()]

        # Create sketch
        cs.accumulateVec(tf.reshape(gradient, [-1]).numpy())

        if momentum > 0:
            # Momentum
            cs.accumulateTable(momentum.numpy()*m_old.numpy())
            m_new = cs.table

            # Store new momentum
            self.momentums[self._index_dict[var_key]].assign(m_new)

        # Error feedback
        cs.table = cs.table*lr.numpy()
        cs.accumulateTable(error.numpy())
        error = cs.table

        # UnSketch with top-k
        k = 10
        if k > d:
            k = d
        delta = cs.unSketch(k=k)

        # Error accumulation
        cs.accumulateVec(tf.reshape(delta, [-1]).numpy())
        sketched_delta = cs.table
        self.error[variable.ref()].assign(error - sketched_delta)

        # Update
        delta = tf.reshape(delta, input_shape)
        variable.assign_add(-delta)

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