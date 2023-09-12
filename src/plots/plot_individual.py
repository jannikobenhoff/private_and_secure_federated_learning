import math

import numpy as np
import matplotlib.pyplot as plt

from src.compressions.Atomo import Atomo
from src.compressions.GradientSparsification import GradientSparsification
from src.compressions.vqSGD import vqSGD
import tensorflow as tf
import os
import seaborn as sns

from src.plots.plot_utils import markers

# from src.utilities.client_data import client_datasets, label_splitter
# from src.utilities.datasets import load_dataset

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


def plot_gspar():
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax2 = ax1.twinx()

    # Generate a test gradient with values from a uniform distribution
    test_gradient = tf.random.uniform(shape=[1000])

    # Create a series of histograms for different repetition parameters
    repetitions = [0.5, 0.1, 0.05, 0.01, 0.005]
    var_values = []
    cr = []
    for i, kappa in enumerate(repetitions):
        gspar = GradientSparsification(k=kappa, max_iter=2)

        compressed_data = gspar.compress(test_gradient, tf.Variable(test_gradient, name="test"), log=True)
        # decompressed = gspar.decompress(compressed_data, tf.Variable(test_gradient, name="test"))
        mean = np.mean(compressed_data["compressed_grads"])
        var = np.sum([np.power((x - mean), 2) for x in compressed_data["compressed_grads"]]) / 999
        var_values.append(var)
        cr.append(gspar.compression_rates[0])
        ax1.scatter([kappa], [var], marker=markers["Gradient Sparsification"], label=f"kappa = {kappa}")
        ax2.scatter([kappa], [cr[-1]], alpha=0.5)

    ax1.plot(repetitions, var_values, color="b", alpha=0.4)
    # ax2.plot(repetitions, cr, color="g", alpha=0.4)
    ax1.set_xlabel('kappa', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_ylabel('Variance', fontsize=8)
    ax2.set_ylabel('Compression Ratio', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)
    plt.show()
    plt.tight_layout()
    # plt.savefig("repetition.pdf")


def plot_atomo():
    fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    # ax2 = ax1.twinx()

    # Generate a test gradient with values from a uniform distribution
    test_gradient = [tf.random.uniform(shape=[100, 50]), tf.random.uniform(shape=[100, 50])]

    # Create a series of histograms for different repetition parameters
    ranks = [1, 2, 3, 4, 6]
    var_values = []
    cr = []
    for i, rank in enumerate(ranks):
        atomo = Atomo(svd_rank=rank, random_sample=False)

        compressed_data = atomo.compress(test_gradient, tf.Variable(test_gradient, name="test"), log=True)
        for i, _ in enumerate(compressed_data["compressed_grads"]):
            s = compressed_data["compressed_grads"][i]["s"]
            u = compressed_data["compressed_grads"][i]["u"]
            vT = compressed_data["compressed_grads"][i]["vT"]
            print(s.shape, u.shape, vT.shape)
        decompressed = atomo.decompress(compressed_data, tf.Variable(test_gradient, name="test"))
        mean = np.mean(decompressed)
        var = np.sum([np.power((x - mean), 2) for x in decompressed]) / 999
        var_values.append(var)
        cr.append(atomo.compression_rates[0])
        # ax1[0].scatter([rank], [var], marker=markers["Gradient Sparsification"], label=f"rank = {rank}")
        # ax1[1].scatter([rank], [cr[-1]], marker=markers["Gradient Sparsification"], label=f"rank = {rank}")
        print(rank, cr[-1])

    # ax1[0].plot(ranks, var_values, color="b", alpha=0.4)
    # ax1[1].plot(ranks, cr, color="g", alpha=0.4)
    # ax1[0].set_xlabel('kappa', fontsize=8)
    # plt.tick_params(axis='both', which='major', labelsize=8)
    # ax1[0].set_ylabel('Variance', fontsize=8)
    # ax1[1].set_ylabel('Compression Ratio', fontsize=8)
    # ax1[0].grid(True, alpha=0.2)
    # ax1[0].legend(fontsize=8)
    # plt.show()
    # plt.tight_layout()
    # plt.savefig("repetition.pdf")


def plot_diri():
    number_clients = 10
    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist",
                                                                                          fullset=100)

    def count_labels_per_client(client_labels, num_classes):
        label_counts_per_client = []

        for labels in client_labels:
            counts = np.bincount(labels.flatten(), minlength=num_classes)
            label_counts_per_client.append(counts)

        return label_counts_per_client

    (split_data, split_labels) = label_splitter(img_train, label_train)

    (client_data, client_labels) = client_datasets(number_clients=number_clients, split_type="dirichlet",
                                                   list_data=split_data, list_labels=split_labels, beta=2)

    label_counts = count_labels_per_client(client_labels, num_classes)

    # Print the counts for each client
    for i, counts in enumerate(label_counts):
        print(f"Client {i}: {sum(counts)}")

    data = np.array(label_counts)

    fig = plt.plot(figsize=(7, 5))
    # Create a heatmap
    sns.heatmap(data, annot=True, fmt="d", cmap='coolwarm')
    plt.xlabel('Class', fontsize=8)
    plt.ylabel('Client', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)

    # plt.title('Samples per Class for Each Client')

    plt.tight_layout()
    # plt.show()

    plt.savefig("dirichlet2.pdf")


def plot_local_iter():
    np.random.seed(100)
    clients = list(range(1, 11))

    local_iter = np.round(np.random.dirichlet([0.5] * 10) * 30)
    local_iter = [2 for i in range(10)]
    print(local_iter)
    fig = plt.plot(figsize=(7, 5))
    plt.bar(clients, local_iter, color='b')

    # Adding labels and title
    plt.xlabel('Client Number', fontsize=8)
    plt.ylabel('Number of Local Iterations', fontsize=8)
    # plt.title('Local Iterations Per Client')
    # plt.xticks(clients)
    yint = range(0, 10, 1)

    plt.yticks(yint)
    # Displaying value labels on each bar
    # for i, val in enumerate(local_iter):
    #     plt.text(i, val + 0.3, str(val), ha='center')

    # Display the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("local_iter_same.pdf")


if __name__ == "__main__":
    plot_atomo()
