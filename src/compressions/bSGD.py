import itertools

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression


class bSGD(Compression):
    def __init__(self, buckets, sparse_buckets, name="BucketSGD"):
        super().__init__(name=name)
        self.buckets = buckets
        self.sparse_buckets = sparse_buckets
        self.compression_rates = []
        self.error = {}

    def build(self, var_list):
        """Initialize optimizer variables.

        bSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        for var in var_list:
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error", initial_value=tf.zeros_like(var)
            )
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        gradient = gradient + self.error[variable.ref()]
        flattened_gradient = tf.reshape(gradient, [-1])
        input_shape = gradient.shape

        ft_np = tf.abs(flattened_gradient).numpy()
        sorted_indices = np.argsort(ft_np)[::-1]

        # Split the sorted indices into 'buckets' number of parts
        partitioned_indices = np.array_split(sorted_indices, self.buckets)
        indices = list(itertools.chain.from_iterable(partitioned_indices))

        sorted_arr = sorted(ft_np, reverse=True)

        # Split the sorted array into 'buckets' number of parts
        partitioned_list = np.array_split(sorted_arr, self.buckets)

        for i, part in enumerate(partitioned_list):
            if self.buckets - (i + 1) < self.sparse_buckets:
                partitioned_list[i] = np.zeros_like(part)
            else:
                partitioned_list[i] = [np.mean(part)] * len(part)

        # partitioned_list = [[np.mean(part)] * len(part) if i < (self.buckets - self.sparse_buckets)
        #                     else np.zeros_like(part)
        #                     for i, part in enumerate(partitioned_list)]

        sparse_gradient = tf.gather(tf.constant(list(itertools.chain.from_iterable(partitioned_list))),
                                    tf.argsort(indices))
        sparse_gradient = sparse_gradient * tf.sign(flattened_gradient)

        if variable.ref() not in self.cr:
            bit_sizes = [
                # (self.buckets - self.sparse_buckets) * len(partitioned_list[-1]) + (
                #     self.buckets - self.sparse_buckets) * 32 * tf.int32.size * 8,
                self.get_bucket_tensor_size_in_bits(sparse_gradient, self.buckets, self.sparse_buckets),
                self.get_sparse_tensor_size_in_bits(sparse_gradient)]

            bit_size = min(bit_sizes)
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / bit_size

            self.compression_rates.append(self.cr[variable.ref()])
            print(np.mean(self.compression_rates))

        sparse_gradient = tf.reshape(sparse_gradient, input_shape)

        # Error Feedback
        error = gradient - sparse_gradient
        self.error[variable.ref()].assign(error)

        return sparse_gradient

    @staticmethod
    def get_bucket_tensor_size_in_bits(tensor, buckets: int, sparse_buckets: int):
        flattened_tensor = tf.reshape(tensor, [-1])
        num_nonzero_entries = tf.math.count_nonzero(flattened_tensor)

        # num_elements = tf.size(flattened_tensor, out_type=tf.float32)
        # num_index_bits = tf.math.ceil(tf.math.log(num_elements) / tf.math.log(2.0))

        num_index_bits = tf.int32.size * 8
        num_value_bits = tf.constant(tensor.dtype.size * 8, dtype=tf.int64)

        total_bits = num_nonzero_entries * num_index_bits + (buckets - sparse_buckets) * 2 * num_value_bits
        return min(tf.cast(tf.maximum(total_bits, 1), dtype=tf.int32),
                   tensor.dtype.size * 8 * np.prod(tensor.shape.as_list()))
