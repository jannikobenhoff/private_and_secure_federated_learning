import numpy as np
import matplotlib.pyplot as plt

from src.compressions.vqSGD import vqSGD
import tensorflow as tf
import os

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.config.run_functions_eagerly(run_eagerly=True)
tf.data.experimental.enable_debug_mode()


def plot_vqsgd():
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax2 = ax1.twinx()

    # Generate a test gradient with values from a uniform distribution
    test_gradient = tf.random.uniform(shape=[1000])

    # Create a series of histograms for different repetition parameters
    repetitions = [10, 25, 50, 100, 300, 700, 1000]
    var_values = []
    cr = []
    for i, rep in enumerate(repetitions):
        vqsgd = vqSGD(repetition=rep)

        compressed_data = vqsgd.compress(test_gradient, tf.Variable(test_gradient, name="test"))
        decompressed = vqsgd.decompress(compressed_data, tf.Variable(test_gradient, name="test"))
        mean = np.mean(decompressed)
        var = np.sum([np.power((x - mean), 2) for x in decompressed]) / 999
        var_values.append(var)
        cr.append((1000 * tf.float32.size * 8) / (tf.float32.size * 8 + rep * np.log2(2 * 1000)))

        ax1.scatter([rep], [var], marker="^", label=f"{rep} repetitions")
        ax2.scatter([rep], [cr[-1]], alpha=0.0)

    ax1.plot(repetitions, var_values, color="b", alpha=0.4)
    # ax2.plot(repetitions, cr, color="g", alpha=0.4)
    plt.xlabel('Repetition', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_ylabel('Variance', fontsize=8)
    ax2.set_ylabel('Compression Ratio', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)
    # plt.show()
    plt.tight_layout()
    plt.savefig("repetition.pdf")


if __name__ == "__main__":
    plot_vqsgd()
