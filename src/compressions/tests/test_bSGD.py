import unittest
from src.compressions.bSGD import *
import tensorflow as tf


def compress(gradient: Tensor, buckets, sparse_buckets) -> Tensor:
    flattened_gradient = tf.reshape(gradient, [-1])
    input_shape = gradient.shape

    ft_np = tf.abs(flattened_gradient).numpy()
    sorted_indices = np.argsort(ft_np)[::-1]

    # Split the sorted indices into 'buckets' number of parts
    partitioned_indices = np.array_split(sorted_indices, buckets)
    indices = list(itertools.chain.from_iterable(partitioned_indices))

    sorted_arr = sorted(ft_np, reverse=True)

    # Split the sorted array into 'buckets' number of parts
    partitioned_list = np.array_split(sorted_arr, buckets)

    for i, part in enumerate(partitioned_list):
        if buckets - (i + 1) < sparse_buckets:
            partitioned_list[i] = np.zeros_like(part)
        else:
            partitioned_list[i] = [np.mean(part)] * len(part)

    # partitioned_list = [[np.mean(part)] * len(part) if i < (self.buckets - self.sparse_buckets)
    #                     else np.zeros_like(part)
    #                     for i, part in enumerate(partitioned_list)]

    sparse_gradient = tf.gather(tf.constant(list(itertools.chain.from_iterable(partitioned_list))),
                                tf.argsort(indices))
    sparse_gradient = sparse_gradient * tf.sign(flattened_gradient)

    sparse_gradient = tf.reshape(sparse_gradient, input_shape)

    return sparse_gradient


def process_tensor(gradient, bucket, sparse_bucket):
    flat_tensor = tf.abs(tf.reshape(gradient, [-1])).numpy()

    bucket = min(bucket, len(flat_tensor))
    bucket_size = max(1, len(flat_tensor) // bucket)

    if sparse_bucket > len(flat_tensor):
        sparse_bucket = len(flat_tensor)
        bucket = sparse_bucket + 1

    # elif bucket - sparse_bucket > len(flat_tensor):
    #     bucket = len(flat_tensor) + sparse_bucket

    indices = np.argpartition(np.abs(flat_tensor.ravel()), -(bucket - sparse_bucket) * bucket_size)[
              -(bucket - sparse_bucket) * bucket_size:]

    output_tensor = np.zeros_like(flat_tensor)

    for i in range(bucket - sparse_bucket):
        start_idx = i * bucket_size
        end_idx = (i + 1) * bucket_size
        output_tensor[indices[start_idx:end_idx]] = np.mean([flat_tensor[indices[start_idx:end_idx]]])

    output_tensor = tf.reshape(output_tensor, gradient.shape)
    output_tensor = output_tensor * tf.sign(gradient)
    return output_tensor


class TestbSGD(unittest.TestCase):
    def test_bsgd(self):
        tf.config.set_visible_devices([], 'GPU')
        tf.config.run_functions_eagerly(run_eagerly=True)
        tf.data.experimental.enable_debug_mode()

        b = bSGD(buckets=3, sparse_buckets=1)

        grad = tf.constant(np.random.rand(10, 100), dtype=tf.float32)
        # grad = tf.constant([1, 2, 8, -4, 5, -6, -7, 3], dtype=tf.float32)
        p = process_tensor(grad, 100, 99)
        # c = compress(grad, 3, 1)

        print(tf.unique(tf.reshape(p, [-1]))[0].numpy())
        print(tf.math.count_nonzero(p))

        grad = tf.constant([1, 2, 8, -4, 5, -6, -7, 3], dtype=tf.float32)
        p = process_tensor(grad, 10000, 9900)

        print(tf.unique(tf.reshape(p, [-1]))[0].numpy())
        print(tf.math.count_nonzero(p))


if __name__ == '__main__':
    unittest.main()
