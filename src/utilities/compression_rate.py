from tensorflow import Tensor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_sparse_tensor_size_in_bits(tensor):
    num_nonzero_entries = tf.math.count_nonzero(tensor)
    num_index_bits = np.ceil(np.log2(len(tf.reshape(tensor, [-1]))))
    num_value_bits = tensor.dtype.size * 8
    return num_nonzero_entries.numpy() * (num_index_bits + num_value_bits) if num_nonzero_entries.numpy() * (
                num_index_bits + num_value_bits) != 0 else 1


def plot_compression_rates(compression_dict):
    fig, ax = plt.subplots()

    table_data = [[name, rate] for name, rate in compression_dict.items()]
    table = plt.table(cellText=table_data,
                      colLabels=["Compression Method", "Compression Rate"],
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
        'Gradient Sparsification (k=0.02)': 225,
        'Natural Compression': 4,
        '1-Bit SGD': 32,
        'Sparse Gradient (R=90%)': 7.4,
        'TernGrad': 16,
        'Top-K (k=10)': 306.3,
        'vqSGD (s=200)': 7.7,
        'EFsignSGD': "-",
        'FetchSGD': "-",
        'MemSGD (k=10)': 306.3
    }

    # plot_compression_rates(compression_dict)
    # print(get_sparse_tensor_size_in_bits(tf.constant([0,0,1,2,3,4], dtype=tf.float32)))
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
