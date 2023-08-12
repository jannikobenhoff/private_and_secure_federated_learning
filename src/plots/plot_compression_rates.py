import ast
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_compression_rates():
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


def plot_compression_metrics(title: str):
    baseline = 'training_SGD_mnist_08_07_00_18.json'
    plot_configs = {
        "gradientsparsification": ["max_iter", "k"],
        "fetchsgd": ["c"],
        "sparsegradient": ["drop_rate"],
        "vqsgd": ["repetition"],
        "topk": ["k"],
        "memsgd": ["top_k"],
        "naturalcompression": [],
        "efsignsgd": [],
        "onebitsgd": []
    }
    params = plot_configs[title]
    marker = itertools.cycle(('s', '+', 'v', 'o', '*'))

    base = open("../results/compression/" + baseline, "r")
    baseline_metrics = json.load(base)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    all_params = [[-100]]
    all_cr = [1]
    all_val_acc = [np.mean(ast.literal_eval(baseline_metrics["val_acc"]))]
    all_val_loss = [ast.literal_eval(baseline_metrics["val_loss"])]
    all_val_acc_plots = [ast.literal_eval(baseline_metrics["val_acc"])]

    for res in os.listdir("../results/compression/" + title):
        file = open("../results/compression/" + title + "/" + res, "r")
        metrics = json.load(file)
        param_value = []
        if len(params) > 0:
            for param in params:
                param_value.append(ast.literal_eval(metrics["args"]["strategy"])[param])
            print(param_value, "lr:", ast.literal_eval(metrics["args"]["strategy"])["learning_rate"])
        else:
            all_params.append(title)
        all_params.append(param_value)
        all_cr.append(np.mean(metrics["compression_rates"]))
        all_val_acc.append(np.mean(ast.literal_eval(metrics["val_acc"])))
        all_val_acc_plots.append(ast.literal_eval(metrics["val_acc"]))
        all_val_loss.append(ast.literal_eval(metrics["val_loss"]))

    if len(params) > 0:
        # Sorting data by parameter
        all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = zip(
            *sorted(zip(all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots)))
        all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = list(all_params), list(all_cr), list(
            all_val_acc), list(all_val_loss), list(all_val_acc_plots)

        all_params = [" ".join([str(p) for p in param_value]) for param_value in all_params]

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
                  zip(all_params, all_cr, all_val_acc) if name != "-100" and name != [-100]]
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
    # plt.savefig("../../figures/methods/" + title + ".pdf", bbox_inches='tight')
    plt.show()


def plot_compare_all():
    def get_all_files_in_directory(root_path):
        all_files = []
        for subdir, dirs, files in os.walk(root_path):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                all_files.append(file_path)
        return all_files

    directory_path = '../results/compression/'
    all_files = get_all_files_in_directory(directory_path)

    all_acc = {}
    all_cr = {}
    all_loss = {}
    for file_path in all_files:
        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        if strat["optimizer"] + " " + strat["compression"] in all_acc:
            if all_acc[strat["optimizer"] + " " + strat["compression"]][1] < np.mean(ast.literal_eval(file["val_acc"])):
                all_acc[strat["optimizer"] + " " + strat["compression"]] = [file["compression_rates"][0],
                                                                            np.mean(ast.literal_eval(file["val_acc"]))]
                all_loss[strat["optimizer"] + " " + strat["compression"]] = ast.literal_eval(file["val_loss"])
        else:
            all_acc[strat["optimizer"] + " " + strat["compression"]] = [file["compression_rates"][0],
                                                                        np.mean(ast.literal_eval(file["val_acc"]))]
            all_loss[strat["optimizer"] + " " + strat["compression"]] = ast.literal_eval(file["val_loss"])

        if strat["optimizer"] + " " + strat["compression"] in all_cr:
            if all_cr[strat["optimizer"] + " " + strat["compression"]][0] < np.mean(file["compression_rates"]):
                all_cr[strat["optimizer"] + " " + strat["compression"]] = [file["compression_rates"][0],
                                                                           np.mean(ast.literal_eval(file["val_acc"]))]
        else:
            all_cr[strat["optimizer"] + " " + strat["compression"]] = [file["compression_rates"][0],
                                                                       np.mean(ast.literal_eval(file["val_acc"]))]

    marker = itertools.cycle(('s', '+', 'v', 'o', '*'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for a in all_acc:
        m = next(marker)
        axes[0].scatter(all_acc[a][0], all_acc[a][1], label=a.replace("none", "") if a[-4:] == "none" else a[4:],
                        marker=m)
        axes[2].scatter(all_cr[a][0], all_cr[a][1], label=a.replace("none", "") if a[-4:] == "none" else a[4:],
                        marker=m)
        axes[1].plot(np.arange(1, 21, 1), all_loss[a], marker=m)

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Max Validation Acc / Compression Rate", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Validation Loss", fontsize=10)
    axes[2].set_title("Validation Acc / Max Compression Rate", fontsize=10)

    axes[2].grid(alpha=0.2)

    table_data = [[name.replace("none", "") if name[-4:] == "none" else name[4:], round(100 * rate[1], 2),
                   round(rate[0], 2)] for name, rate in all_acc.items()]

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    table = axes[3].table(cellText=table_data,
                          colLabels=["Method", "Val Acc", "Compression Rate"],
                          cellLoc='center',
                          loc='center')

    axes[3].axis('off')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(10)
            cell._text.set_weight('bold')
    plt.tight_layout()
    # plt.savefig("../../figures/compare_all.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_compression_metrics("vqsgd")

    # plot_compare_all()

    # plot_compression_rates()
