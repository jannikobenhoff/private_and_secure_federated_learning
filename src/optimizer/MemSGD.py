import numpy as np
import tensorflow as tf
from keras.optimizers.optimizer import Optimizer
from tensorflow import Tensor


class MemSGD(Optimizer):
    def __init__(self, learning_rate, top_k: int = None, rand_k: int = None, name="MemSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)

        if rand_k is not None and top_k is not None:
            raise "Please only select top-k or random-k sparsification."
        elif rand_k is None and top_k is None:
            raise "Please select a sparsification method, top-k or random-k."
        self.top_k = top_k
        self.rand_k = rand_k
        self.compression_rates = []
        self.cr = {}

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        MemSGD optimizer has one variable `momentum`

        Args:
          var_list: list of model variables to build MemSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return

        if clients == 1:
            # Local Setup
            self.memory = []

            for var in var_list:
                self.memory.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="m", initial_value=tf.zeros(shape=var.shape)
                    )
                )
        else:
            self.memory = {}
            # Federated Setup
            for client_id in range(1, clients + 1):
                self.memory[str(client_id)] = []
                for var in var_list:
                    self.memory[str(client_id)].append(
                        self.add_variable_from_reference(
                            model_variable=var, variable_name="m", initial_value=tf.zeros(shape=var.shape)
                        )
                    )
        self._built = True

    def update_step(self, gradient: Tensor, variable, lr) -> Tensor:
        self.lr = lr
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        var_key = self._var_key(variable)

        m = self.memory[self._index_dict[var_key]]
        if gradient.dtype != variable.dtype:
            gradient = tf.cast(gradient, dtype=variable.dtype)

        if self.top_k is None:
            g = self.rand_k_sparsification(input_tensor=m + lr * gradient,
                                           k=self.rand_k)
        else:
            g = self.top_k_sparsification(input_tensor=m + lr * gradient,
                                          k=self.top_k)
        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                g)
            self.compression_rates.append(self.cr[variable.ref()])

        self.memory[self._index_dict[var_key]].assign(m + lr * gradient - g)

        return g

    def federated_compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int, lr):
        self.lr = lr
        lr = tf.cast(self.lr, variables[0].dtype.base_dtype)
        quantized_gradients = []
        for i, gradient in enumerate(gradients):

            var_key = self._var_key(variables[i])

            m = self.memory[str(client_id)][self._index_dict[var_key]]

            if gradient.dtype != variables[i].dtype:
                gradient = tf.cast(gradient, dtype=variables[i].dtype)

            if self.top_k is None:
                g = self.rand_k_sparsification(input_tensor=m + lr * gradient,
                                               k=self.rand_k)
            else:
                g = self.top_k_sparsification(input_tensor=m + lr * gradient,
                                              k=self.top_k)
            if variables[i].ref() not in self.cr:
                self.cr[variables[i].ref()] = gradient.dtype.size * 8 * np.prod(
                    gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                    g)
                self.compression_rates.append(self.cr[variables[i].ref()])

            self.memory[str(client_id)][self._index_dict[var_key]].assign(m + lr * gradient - g)
            quantized_gradients.append(g)
        return {
            'compressed_grad': quantized_gradients,
            'decompress_info': None
        }

    def federated_decompress(self, client_data, variables, lr):
        for i, gradient in client_data["compressed_grad"]:
            client_data["compressed_grad"] = gradient / lr
        return client_data["compressed_grad"]

    @staticmethod
    def top_k_sparsification(input_tensor: Tensor, k: int) -> Tensor:
        """
        Returns a sparse tensor of the input tensor, with the top k elements.
        k = 2: [1, 2, 3] -> [0, 2, 3]
        """
        input_shape = input_tensor.shape
        flattened_tensor: Tensor = tf.reshape(input_tensor, [-1])

        if tf.size(flattened_tensor) <= k:
            return input_tensor

        ft_np = tf.abs(flattened_tensor).numpy()

        indices = np.argpartition(np.abs(ft_np.ravel()), -k)[-k:]
        mask = tf.zeros(flattened_tensor.shape, dtype=flattened_tensor.dtype)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1),
                                           tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = flattened_tensor * mask

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
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1),
                                           tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = tf.math.multiply(flattened_tensor, mask)
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor

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
    def get_sparse_tensor_size_in_bits(tensor):
        flattened_tensor = tf.reshape(tensor, [-1])
        num_nonzero_entries = tf.math.count_nonzero(flattened_tensor)

        num_index_bits = tf.int32.size * 8
        num_value_bits = tf.constant(tensor.dtype.size * 8, dtype=tf.int64)

        total_bits = num_nonzero_entries * (num_index_bits + num_value_bits)
        return min(tf.cast(tf.maximum(total_bits, 1), dtype=tf.int32),
                   tensor.dtype.size * 8 * np.prod(tensor.shape.as_list()))
