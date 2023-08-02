from tensorflow import Tensor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_compression_rate(uncompressed: Tensor, compressed: Tensor):
    # PSNR = 10 * log10(max_value ^ 2 / mean_squared_error)
    # print(np.sqrt(np.sum(np.power(compressed.numpy(), 2))), np.sqrt(np.sum(np.power(uncompressed.numpy(), 2))))
    compressed = tf.sparse.from_dense(compressed).values

    # print("Un:", len(bytearray(uncompressed.numpy())))
    # print("Co:", len(bytearray(compressed.numpy())))
    # print(uncompressed.dtype.size,  compressed.dtype.size)
    compressed = tf.sparse.from_dense(compressed).values
    original_size = np.prod(uncompressed.shape.as_list()) * uncompressed.dtype.size
    compressed_size = np.prod(compressed.shape.as_list()) * compressed.dtype.size
    compression_ratio = np.divide(original_size, compressed_size)
    # print("Compression Ratio:", compression_ratio)
    return compression_ratio


def get_sparse_tensor_size_in_bits(tensor):
    num_nonzero_entries = tf.math.count_nonzero(tensor)
    num_index_bits = 32
    num_value_bits = tensor.dtype.size * 8
    return num_nonzero_entries.numpy() * (num_index_bits + num_value_bits) if num_nonzero_entries.numpy() * (
                num_index_bits + num_value_bits) != 0 else 1


def plot_compression_rates(compression_dict):
    fig, ax = plt.subplots()

    table_data = [[name, rate] for name, rate in compression_dict.items()]
    table = plt.table(cellText=table_data,
                      colLabels=["Compression Method", "Compression Ratio"],
                      cellLoc='center',
                      loc='center')

    ax.axis('off')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(10)
            cell._text.set_weight('bold')

    #plt.show()
    plt.savefig("compression_rates.pdf")


if __name__ == "__main__":
    compression_dict = {
        'SGD': 1,
        'Gradient Sparsification (k=0.02)': 205,
        'Natural Compression': 4,
        '1-Bit SGD': 32,
        'Sparse Gradient (R=90%)': 4.67,
        'TernGrad': 16,
        'Top-K (k=10)': 222.15,
        'vqSGD (s=200)': 7.7,
        'EFsignSGD': "-",
        'FetchSGD': "-",
        'MemSGD (k=10)': 222.15,
    }

    plot_compression_rates(compression_dict)
