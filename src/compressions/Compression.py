import abc

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from typing import Final


class Compression:
    def __init__(self, name: str):
        self.name = name
        self._variables = []
        self.compression_rates = []
        self.cr = {}
        self._built = False

    @abc.abstractmethod
    def build(self, var_list, clients=1):
        """Initialize the optimizer's variables, such as momemtum variables.

        This function has to be implemented by subclass optimizers, and subclass
        optimizers need to call `super().build(var_list)`.

        Args:
          var_list: List of model variables to build optimizers on. For example,
            SGD optimizer with momentum will store one momentum variable
            corresponding to each model variable.
          clients: Number of clients in federated learning
        """
        if getattr(self, "_built", False):
            return
        self._build_index_dict(var_list)

    def add_variable_from_reference(
            self, model_variable, variable_name, shape=None, initial_value=None
    ):
        """Create an optimizer variable from model variable.

        Create an optimizer variable based on the information of model variable.
        For example, in SGD optimizer momemtum, for each model variable, a
        corresponding momemtum variable is created of the same shape and dtype.

        Args:
          model_variable: tf.Variable. The corresponding model variable to the
            optimizer variable to be created.
          variable_name: String. The name prefix of the optimizer variable to be
            created. The create variables name will follow the pattern
            `{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
          shape: List or Tuple, defaults to None. The shape of the optimizer
            variable to be created. If None, the created variable will have the
            same shape as `model_variable`.
          initial_value: A Tensor, or Python object convertible to a Tensor,
            defaults to None. The initial value of the optimizer variable, if
            None, the initial value will be default to 0.

        Returns:
          An optimizer variable.
        """
        if initial_value is None:
            if shape is None:
                if model_variable.shape.rank is None:
                    # When the rank is None, we cannot get a concrete
                    # `model_variable.shape`, we use dynamic shape.
                    initial_value = tf.zeros_like(
                        model_variable, dtype=model_variable.dtype
                    )
                else:
                    # We cannot always use `zeros_like`, because some cases
                    # the shape exists while values don't.
                    initial_value = tf.zeros(
                        model_variable.shape, dtype=model_variable.dtype
                    )
            else:
                initial_value = tf.zeros(shape, dtype=model_variable.dtype)
        variable = tf.Variable(
            initial_value=initial_value,
            name=f"{variable_name}/{model_variable._shared_name}",
            dtype=model_variable.dtype,
            trainable=False,
        )
        self._variables.append(variable)
        return variable

    def compress(self, grads: list[Tensor], variables):
        raise NotImplementedError("Subclasses must implement compress method")

    def decompress(self, compressed_data, variables):
        raise NotImplementedError("Subclasses must implement decompress method")

    @staticmethod
    def top_k_sparsification(input_tensor: Tensor, k: int) -> Tensor:
        """
        Returns a sparse tensor of the input tensor, with the top k elements.
        k = 2: [1, 2, 3] -> [0, 2, 3]
        """
        # k = tf.cast(k, tf.int32.base_dtype)

        input_shape: Final = input_tensor.shape
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
    def get_sparse_tensor_size_in_bits(tensor):
        flattened_tensor = tf.reshape(tensor, [-1])
        num_nonzero_entries = tf.math.count_nonzero(flattened_tensor)

        # int32 -> Index 0 -> Index 2 * 2,147,483,647
        # int64 > 2 * 2,147,483,647
        num_index_bits = tf.int32.size * 8
        num_value_bits = tf.constant(tensor.dtype.size * 8, dtype=tf.int64)

        total_bits = num_nonzero_entries * (num_index_bits + num_value_bits)
        return min(tf.cast(tf.maximum(total_bits, 1), dtype=tf.int32),
                   tensor.dtype.size * 8 * np.prod(tensor.shape.as_list()))
