import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class FetchSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, momentum: float, name="FetchSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.num_has = 10
        self.num_counters = 3

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
                    model_variable=var, variable_name="m"
                )
            )
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error"
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]
        error = self.error[variable.ref()]

        sketch = self.count_sketch(gradient, self.num_has, self.num_counters)

        m_new = self.momentum * m + sketch
        self.momentums[self._index_dict[var_key]].assign(m_new)

        error = lr * m_new + error
        delta = self.top_k_sparsification(self.uncount_sketch(error,
                                                              self.num_has,
                                                              self.num_counters),
                                          10)

        self.error[variable.ref()].assign(error - self.count_sketch(delta, self.num_has, self.num_counters))
        variable.assign_add(-delta)

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
        import numpy as np
        c = 5
        x = input_tensor.numpy()
        n = len(x)
        s = np.random.choice([-1, 1], size=n)
        h = np.random.randint(0, c, size=n)
        y = np.zeros(c, dtype=np.float32)
        for i in range(n):
            y[h[i]] += s[i] * x[i]
        return y


    @staticmethod
    def uncount_sketch(counts: tf.Tensor, num_hash_functions: int, num_counters: int):
        import numpy as np
        c = 5
        n = 5
        y = counts
        s = np.random.choice([-1, 1], size=n)
        h = np.random.randint(0, c, size=n)
        n = len(y)
        x_hat = np.zeros(n, dtype=np.float32)
        for i in range(n):
            x_hat[i] = s[i] * y[h[i]]
        return x_hat


    @staticmethod
    def top_k_sparsification(input_tensor: Tensor, k: int) -> Tensor:
        """
        Returns a sparse tensor of the input tensor, with the top k elements.
        k = 2: [1, 2, 3] -> [0, 2, 3]
        """
        input_shape = input_tensor.shape
        flattened_tensor: Tensor = tf.reshape(input_tensor, [-1])

        if tf.size(flattened_tensor) < k:
            return input_tensor

        abs_tensor = tf.abs(flattened_tensor)
        indices = tf.math.top_k(abs_tensor, k).indices

        mask = tf.zeros(flattened_tensor.shape)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1),
                                           tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = tf.math.multiply(flattened_tensor, mask)
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor
