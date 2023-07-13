import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class FetchSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, momentum, name="FetchSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables.

        FetchSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build FetchSGD variables on.
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

        # update residual error
        self.error[variable.ref()].assign()

        # update iterate
        variable.assign_add()

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
    def count_sketch(input_tensor: Tensor, num_hash_functions: int, num_counters: int) -> Tensor:
        """
        Args:
          input_tensor: The tensor to be counted.
          num_hash_functions: The number of hash functions to use.
          num_counters: The number of counters to use.

        Returns:
          A TensorFlow tensor that contains the estimated frequencies of the data.
        """

        hashes = tf.random.uniform([num_hash_functions], minval=0, maxval=num_counters, dtype=tf.int32)
        counts = tf.reduce_min(tf.gather(input_tensor, hashes), axis=0)
        return counts



