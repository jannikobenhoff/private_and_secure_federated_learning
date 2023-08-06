import ast
import json

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


def plot_compression_metrics(results: list, params: list, baseline: str):
    marker = itertools.cycle(('s', '+', 'v', 'o', '*'))

    base = open("../results/compression/" + baseline, "r")
    baseline_metrics = json.load(base)
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].plot(np.arange(0, len(ast.literal_eval(baseline_metrics["val_acc"])[0])),
                 ast.literal_eval(baseline_metrics["val_acc"])[0], label=f"baseline")
    axes[3].plot(np.arange(0, len(ast.literal_eval(baseline_metrics["val_loss"])[0])),
                 ast.literal_eval(baseline_metrics["val_loss"])[0], label=f"baseline")
    all_params = []
    all_cr = []
    all_val_acc = []
    for res in results:
        for param in params:
            file = open("../results/compression/" + res, "r")
            metrics = json.load(file)
            param_value = ast.literal_eval((metrics["args"]["strategy"]))[param]
            cr = np.mean(metrics["compression_rates"])
            print(param_value, cr, np.mean(ast.literal_eval(metrics["val_acc"])[0]))
            all_params.append(param_value)
            all_cr.append(cr)
            val_acc = ast.literal_eval(metrics["val_acc"])[0]
            val_loss = ast.literal_eval(metrics["val_loss"])[0]
            all_val_acc.append(val_acc)
            m = next(marker)
            axes[0].plot(np.arange(0, len(val_acc)),
                         val_acc, marker=m, label=f"{param}: {param_value}")
            axes[3].plot(np.arange(0, len(val_loss)),
                         val_loss, marker=m, label=f"{param}: {param_value}")
            axes[1].scatter([param_value], [cr], label=f"{param}: {param_value}", marker=m)
            axes[2].scatter(cr, np.mean(val_acc), marker=m, label=f"{param}: {param_value}")



    axes[2].set_xlabel('Compression Rate')
    axes[2].set_ylabel('Validation Accuracy')
    axes[2].set_title('Validation Accuracy vs Compression Rate', fontsize=10)
    axes[2].legend()
    axes[3].legend()

    axes[1].plot(all_params, all_cr, alpha=0.2)
    axes[1].legend()
    axes[1].set_title("Compression Rate", fontsize=10)

    axes[0].set_title("Validation Accuracy", fontsize=10)
    axes[0].legend()
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
    plot_compression_metrics(["training_SGD_TopK_mnist_08_05_22_45.json",
                              "training_SGD_TopK_mnist_08_05_22_48.json",
                              "training_SGD_TopK_mnist_08_05_23_07.json",
                              "training_SGD_TopK_mnist_08_05_23_10.json",
                              "training_SGD_TopK_mnist_08_05_22_52.json",

                              ],
                             ["k"],
                             'training_SGD_mnist_08_05_22_42.json')

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
