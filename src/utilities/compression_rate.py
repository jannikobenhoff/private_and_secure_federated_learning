from tensorflow import Tensor
import tensorflow as tf
import numpy as np


def get_compression_rate(uncompressed: Tensor, compressed: Tensor):
    # PSNR = 10 * log10(max_value ^ 2 / mean_squared_error)
    print(np.sqrt(np.sum(np.power(compressed.numpy(), 2))), np.sqrt(np.sum(np.power(uncompressed.numpy(), 2))))
    compressed = tf.sparse.from_dense(compressed).values

    #print("Un:", len(bytearray(uncompressed.numpy())))
    #print("Co:", len(bytearray(compressed.numpy())))
    #print(uncompressed.dtype.size,  compressed.dtype.size)
    compressed = tf.sparse.from_dense(compressed).values
    original_size = np.prod(uncompressed.shape.as_list()) * uncompressed.dtype.size
    compressed_size = np.prod(compressed.shape.as_list()) * compressed.dtype.size
    compression_ratio = np.divide(original_size, compressed_size)
    # print("Compression Ratio:", compression_ratio)
    return compression_ratio
