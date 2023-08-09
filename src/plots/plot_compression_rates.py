import ast
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import itertools


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

    # plt.show()
    plt.savefig("compression_rates.pdf")


def plot_compression_metrics(params: list, title: str, baseline: str):
    marker = itertools.cycle(('s', '+', 'v', 'o', '*'))

    base = open("../results/compression/" + baseline, "r")
    baseline_metrics = json.load(base)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    # Gathering data
    all_params = [-100]
    all_cr = [1]
    all_val_acc = [np.mean(ast.literal_eval(baseline_metrics["val_acc"]))]
    all_val_loss = [ast.literal_eval(baseline_metrics["val_loss"])]
    all_val_acc_plots = [ast.literal_eval(baseline_metrics["val_acc"])]


    for res in sorted(os.listdir("../results/compression/" + title + "/")):
        for param in params:
            file = open("../results/compression/" + title + "/" + res, "r")
            metrics = json.load(file)
            param_value = ast.literal_eval(metrics["args"]["strategy"])[param]
            all_params.append(param_value)
            all_cr.append(np.mean(metrics["compression_rates"]))
            all_val_acc.append(np.mean(ast.literal_eval(metrics["val_acc"])))
            all_val_acc_plots.append(ast.literal_eval(metrics["val_acc"]))
            all_val_loss.append(ast.literal_eval(metrics["val_loss"]))
            print(param_value, "lr:", ast.literal_eval(metrics["args"]["strategy"])["learning_rate"])

    # Sorting data by parameter
    all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = zip(
        *sorted(zip(all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots)))
    all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = list(all_params), list(all_cr), list(all_val_acc), list(all_val_loss), list(all_val_acc_plots)
    # all_params.append(["baseline"])
    # all_cr.append([1])
    # all_val_acc.append([round(100 * np.mean(ast.literal_eval(baseline_metrics["val_acc"])), 2)])
    # all_val_loss.append([ast.literal_eval(baseline_metrics["val_loss"])])
    # all_val_acc_plots.append([ast.literal_eval(baseline_metrics["val_acc"])])
    # Plotting data
    for param, cr, val_acc, val_loss, val_acc_plot, m in zip(all_params[1:], all_cr[1:], all_val_acc[1:],
                                                             all_val_loss[1:], all_val_acc_plots[1:], marker):
        axes[0].plot(np.arange(0, len(val_acc_plot)), val_acc_plot, marker=m, label=f"{param}")
        axes[3].plot(np.arange(0, len(val_loss)), val_loss, marker=m, label=f"{param}")
        axes[2].scatter(cr, val_acc, marker=m, label=f"{param}")

    # Plotting baseline
    axes[0].plot(np.arange(0, len(all_val_acc_plots[0])), all_val_acc_plots[0], label="baseline")
    axes[3].plot(np.arange(0, len(all_val_loss[0])), all_val_loss[0], label="baseline")
    axes[2].scatter(all_cr[0], all_val_acc[0], label="baseline")

    # Table data
    table_data = [[name, round(rate, 2), str(round(100 * acc, 2)) + " %"] for name, rate, acc in
                  zip(all_params, all_cr, all_val_acc) if name != -100]
    table_data.append(["baseline", "1", str(round(100 * all_val_acc[0], 2)) + " %"])
    table = axes[1].table(cellText=table_data,
                          colLabels=[params, "Compression Rate", "Val Accuracy"],
                          cellLoc='center',
                          loc='center')

    axes[1].axis('off')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(9)
            cell._text.set_weight('bold')

    axes[2].set_xlabel('Compression Rate')
    axes[2].set_ylabel('Validation Accuracy')
    axes[2].set_title('Validation Accuracy vs Compression Rate', fontsize=10)
    axes[2].legend()

    # axes[3].legend()
    axes[3].set_title("Validation Loss", fontsize=10)

    # all_params, all_cr = zip(*sorted(zip(all_params, all_cr)))
    # axes[1].plot(all_params, all_cr, alpha=0.2)
    # axes[1].legend()
    # axes[1].set_title("Compression Rate", fontsize=10)

    axes[0].set_title("Validation Accuracy", fontsize=10)
    axes[0].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    - baseline model mit l2 regularization
    - training over epochs mit k-val und early stopping
    - 5 different param sets
    - 2 plots
        1. CR - Val Acc
        2. CR - Val Loss
    """
    # plot_compression_metrics(["max_iter"],
    #                          "GradientSparsification",
    #                          'training_SGD_mnist_08_07_00_18.json')

    plot_compression_metrics(["repetition"],
                             "vqsgd",
                             'training_SGD_mnist_08_07_00_18.json')

    # plot_compression_metrics(["k"],
    #                          "topk",
    #                          'training_SGD_mnist_08_07_00_18.json')

    # plot_compression_metrics(["top_k"],
    #                          "memsgd",
    #                          'training_SGD_mnist_08_07_00_18.json')


    compression_dict = {
        'SGD': 1,
        'Gradient Sparsification (k=0.02)': 225,
        'Natural Compression': 4,
        '1-Bit SGD': 32,
        'Sparse Gradient (R=90%)': 7.4,
        'TernGrad': np.log2(3),
        'Top-K (k=10)': 306.3,
        'vqSGD (s=200)': 7.7,
        'EFsignSGD': 32,
        'FetchSGD': "1-50",
        'MemSGD (k=10)': 306.3
    }

    # plot_compression_rates(compression_dict)
