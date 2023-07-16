import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class MemSGD(optimizer.Optimizer):
    def __init__(self, learning_rate, top_k: int = None, rand_k: int= None, name="MemSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)

        if rand_k is not None and top_k is not None:
            raise "Please only select top-k or random-k sparsification."
        elif rand_k is None and top_k is None:
            raise "Please select a sparsification method, top-k or random-k."
        self.top_k = top_k
        self.rand_k = rand_k


    def build(self, var_list):
        """Initialize optimizer variables.

        MemSGD optimizer has one variable `momentum`

        Args:
          var_list: list of model variables to build MemSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.memory = []
        for var in var_list:
            self.memory.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m", initial_value=tf.zeros(shape=var.shape)
                )
            )
        self._built = True

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        var_key = self._var_key(variable)
        m = self.memory[self._index_dict[var_key]]

        if self.top_k is None:
            g = self.rand_k_sparsification(input_tensor=m + lr * gradient,
                                           k=self.rand_k)
        else:
            g = self.top_k_sparsification(input_tensor=m + lr * gradient,
                                          k=self.top_k)

        self.memory[self._index_dict[var_key]].assign(m+lr*gradient-g)
        variable.assign_add(-g)

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
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1), tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = tf.math.multiply(flattened_tensor, mask)
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor

    @staticmethod
    def rand_k_sparsification(input_tensor: Tensor, k: int) -> Tensor:
        """
        Returns a sparse tensor with random k elements != 0.
        """
        input_shape = input_tensor.shape
        flattened_tensor = tf.reshape(input_tensor, [-1])
        num_elements = input_tensor.shape.num_elements()

        indices = tf.random.shuffle(tf.range(num_elements))[:k]
        mask = tf.zeros_like(flattened_tensor)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1), tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = tf.math.multiply(flattened_tensor, mask)
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor
