import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow import Tensor

from .Compression import Compression


class MemSGD(Compression):
    def __init__(self, learning_rate, top_k: int = None, rand_k: int = None, name="MemSGD"):
        # super().__init__(name=name)
        # self._learning_rate = self._build_learning_rate(learning_rate)

        super().__init__(name)
        if rand_k is not None and top_k is not None:
            raise "Please only select top-k or random-k sparsification."
        elif rand_k is None and top_k is None:
            raise "Please select a sparsification method, top-k or random-k."
        self.top_k = top_k
        self.rand_k = rand_k
        self.compression_rates = []
        self.cr = {}
        self.name = "MemSGD"

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        MemSGD optimizer has one variable `momentum`

        Args:
          var_list: list of model variables to build MemSGD variables on.
        """
        # super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return

        self.memory = {}
        for client_id in range(1, clients + 1):
            self.memory[str(client_id)] = tf.Variable(tf.zeros(shape=tf.reduce_sum([tf.size(var) for var in var_list])))
        self._built = True

    def compress(self, grads: list[Tensor], variables, client_id, lr):
        flattened_grads = [tf.reshape(grad, [-1]) for grad in grads]
        gradient = tf.concat(flattened_grads, axis=0)

        m = self.memory[str(client_id)]

        if self.top_k is None:
            sparse_gradient = self.rand_k_sparsification(input_tensor=m + lr * gradient,
                                                         k=self.rand_k)
        else:
            sparse_gradient = self.top_k_sparsification(input_tensor=m + lr * gradient,
                                                        k=self.top_k)

        self.memory[str(client_id)].assign(m + lr * gradient - sparse_gradient)

        if variables[0].ref() not in self.cr:
            self.cr[variables[0].ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                sparse_gradient)
            self.compression_rates.append(self.cr[variables[0].ref()])
            self.compression_rates = [np.mean(self.compression_rates)]

        compressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(sparse_gradient[start: start + size])
            # Divided by lr because lr will be multiplied again later in keras process
            compressed_grads.append(tf.reshape(segment, var.shape) / lr)
            start += size

        return {
            "compressed_grads": compressed_grads,
            "decompress_info": None,
            "needs_decompress": False
        }

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
