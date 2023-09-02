import ast
import json
import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import itertools

names = {
    "sgd terngrad": "TernGrad",
    "terngrad": "TernGrad",
    "sgd naturalcompression": "Natural Compression",
    "naturalcompression": "Natural Compression",
    "sgd onebitsgd": "1-Bit SGD",
    "onebitsgd": "1-Bit SGD",
    "sgd sparsegradient": "Sparse Gradient",
    "sparsegradient": "Sparse Gradient",
    "sgd gradientsparsification": "Gradient Sparsification",
    "gradientsparsification": "Gradient Sparsification",
    "memsgd": "Sparsified SGD with Memory",
    "atomo": "Atomic Sparsification",
    "sgd atomo": "Atomic Sparsification",
    "efsignsgd": "EF-SignSGD",
    "fetchsgd": "FetchSGD",
    "vqsgd": "vqSGD",
    "sgd vqsgd": "vqSGD",
    "topk": "TopK",
    "sgd topk": "TopK",
    "sgd ": "SGD",
    "sgd": "SGD",
    "sgd none": "SGD",
    "sgd bsgd": "BucketSGD",
    "bsgd": "BucketSGD"
}


def get_all_files_in_directory(root_path):
    all_files = []
    for subdir, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            all_files.append(file_path)
    return all_files


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


def plot_compression_metrics(title: str, parent_folder: str, baseline):
    # baseline = 'training_SGD_mnist_08_07_00_18.json'
    plot_configs = {
        "gradientsparsification": ["max_iter", "k"],
        "fetchsgd": ["c"],
        "sparsegradient": ["drop_rate"],
        "vqsgd": ["repetition"],
        "topk": ["k"],
        "memsgd": ["top_k"],
        "naturalcompression": [],
        "efsignsgd": [],
        "onebitsgd": [],
        "bsgd": ["buckets", "sparse_buckets"],
        "bsgd2": ["buckets", "sparse_buckets"],
        "terngrad": [],
        "atomo": ["svd_rank"],
        "bucketsgd": ["buckets", "sparse_buckets"],
        "bsgd": ["buckets", "sparse_buckets"],
        "sgd": []
    }
    params = plot_configs[title.lower()]
    marker = itertools.cycle(('s', '+', 'v', 'o', '*'))

    base = open(f"../results/compression/{parent_folder}/" + baseline, "r")
    baseline_metrics = json.load(base)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    all_params = [[-100]]
    all_cr = [1]
    all_val_acc = [np.mean(ast.literal_eval(baseline_metrics["val_acc"]))]
    all_val_loss = [ast.literal_eval(baseline_metrics["val_loss"])]
    all_val_acc_plots = [ast.literal_eval(baseline_metrics["val_acc"])]

    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    for file in all_files:  # os.listdir(f"../results/compression/{parent_folder}/"):
        if title == "bsgd":
            title2 = "BucketSGD"
        else:
            title2 = title
        if title2 not in file:
            continue
        print(file)

        file = open(file, "r")
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

    # all_val_acc_plots = average_and_replicate(all_val_acc_plots)
    # all_val_loss = average_and_replicate(all_val_loss)

    if len(params) > 0:
        # Sorting data by parameter
        all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = zip(
            *sorted(zip(all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots)))
        all_params, all_cr, all_val_acc, all_val_loss, all_val_acc_plots = list(all_params), list(all_cr), list(
            all_val_acc), list(all_val_loss), list(all_val_acc_plots)

        all_params = [" ".join([str(p) for p in param_value]) for param_value in all_params]

    for param, cr, val_acc, val_loss, val_acc_plot, m in zip(all_params[1:], all_cr[1:], all_val_acc[1:],
                                                             all_val_loss[1:], all_val_acc_plots[1:], marker):
        axes[0].plot(np.arange(0, len(val_acc_plot)), val_acc_plot, marker=m, label=f"{param}", markersize=4)
        axes[3].plot(np.arange(0, len(val_loss)), val_loss, marker=m, label=f"{param}")
        axes[2].plot(cr, val_acc, marker=m, label=f"{param}")
        print(val_acc_plot)
    axes[2].plot(all_cr[1:], all_val_acc[1:], c="black", alpha=0.2)  # , marker=m, label=f"{param}")

    # Plotting baseline
    m = next(marker)
    c = "black"
    axes[0].plot(np.arange(0, len(all_val_acc_plots[0])), all_val_acc_plots[0], label="baseline", marker=m, c=c)
    axes[3].plot(np.arange(0, len(all_val_loss[0])), all_val_loss[0], label="baseline", marker=m, c=c)
    axes[2].scatter(all_cr[0], all_val_acc[0], label="baseline", marker=m, c=c)

    # Table data
    table_data = [[name, round(rate, 2), str(round(100 * acc, 2)) + " %"] for name, rate, acc in
                  zip(all_params, all_cr, all_val_acc) if name != "-100" and name != [-100]]
    table_data.append(["baseline", "1", str(round(100 * all_val_acc[0], 2)) + " %"])
    table = axes[1].table(cellText=table_data,
                          colLabels=[params, "Compression Ratio", "Test Accuracy"],
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
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Test Accuracy vs Compression Rate', fontsize=10)
    axes[2].legend()

    # axes[3].legend()
    axes[3].set_title("Test Loss", fontsize=10)
    axes[3].set_yscale("log")

    # all_params, all_cr = zip(*sorted(zip(all_params, all_cr)))
    # axes[1].plot(all_params, all_cr, alpha=0.2)
    # axes[1].legend()
    # axes[1].set_title("Compression Rate", fontsize=10)

    axes[0].set_title("Validation Accuracy", fontsize=10)
    axes[0].legend()
    plt.suptitle(names[title.lower()])
    plt.tight_layout()
    # plt.savefig("../../figures/methods/" + title + ".pdf", bbox_inches='tight')
    plt.show()


def mean_of_arrays_with_padding(arr1, arr2):
    # Determine the length of the longer array
    max_len = max(len(arr1), len(arr2))

    # Pad the shorter array to the length of the longer array
    if len(arr1) < max_len:
        padding = arr2[len(arr1):]
        arr1_padded = np.concatenate((arr1, padding))
        arr2_padded = arr2
    elif len(arr2) < max_len:
        padding = arr1[len(arr2):]
        arr2_padded = np.concatenate((arr2, padding))
        arr1_padded = arr1
    else:
        arr1_padded, arr2_padded = arr1, arr2

    # Compute the mean
    mean_arr = (arr1_padded + arr2_padded) / 2
    return mean_arr


def plot_compare_all(parent_folder: str, bsgd: bool):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        if "Bucket" in file_path and not bsgd:
            continue
        # if not "MEM" in file_path:
        #     continue
        file = open(file_path, "r")
        print(file)
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        train_acc = np.array(ast.literal_eval(file["training_acc"]))
        train_loss = np.array(ast.literal_eval(file["training_loss"]))
        val_acc = np.array(ast.literal_eval(file["val_acc"]))
        val_loss = np.array(ast.literal_eval(file["val_loss"]))
        cr = file["compression_rates"][0]

        # print(strat_key, np.max(val_acc))
        if lean_strat_key not in metrics:
            metrics[lean_strat_key] = {}

        if strat_key not in metrics[lean_strat_key]:
            metrics[lean_strat_key][strat_key] = {}
            metrics[lean_strat_key][strat_key]["train_acc"] = train_acc
            metrics[lean_strat_key][strat_key]["train_loss"] = train_loss
            metrics[lean_strat_key][strat_key]["val_acc"] = val_acc
            metrics[lean_strat_key][strat_key]["val_loss"] = val_loss
            metrics[lean_strat_key][strat_key]["cr"] = cr
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.max(val_acc)

        else:
            metrics[lean_strat_key][strat_key]["train_acc"] = mean_of_arrays_with_padding(train_acc,
                                                                                          metrics[lean_strat_key][
                                                                                              strat_key]["train_acc"])
            metrics[lean_strat_key][strat_key]["train_loss"] = mean_of_arrays_with_padding(train_loss,
                                                                                           metrics[lean_strat_key][
                                                                                               strat_key]["train_loss"])
            metrics[lean_strat_key][strat_key]["val_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                        metrics[lean_strat_key][
                                                                                            strat_key]["val_acc"])
            metrics[lean_strat_key][strat_key]["val_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                         metrics[lean_strat_key][
                                                                                             strat_key]["val_loss"])
            metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    marker = itertools.cycle(('+', 'v', 'o', '*'))
    colors = itertools.cycle(
        ('r', 'g', "#32CD32", 'y', 'm', 'c', 'grey', 'orange', 'pink', "#D2691E", 'b', "#FFD700", "#a6bddb"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    table_data = []
    for method in metrics:
        label_name = names[method.lower().replace(" none", "")]

        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))
        best_param = max(metrics[method].items(), key=lambda x: x[1]['max_val_acc'])
        best_param_metrics = best_param[1]

        table_data.append(
            [label_name, round(100 * best_param_metrics["max_val_acc"], 2), round(best_param_metrics["cr"], 1)])
        m = next(marker)
        c = next(colors)

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        axes[0].plot(np.arange(1, len(best_param_metrics["train_acc"]) + 1, 1), best_param_metrics["train_acc"],
                     markersize=4, color=c,  # marker=m,
                     label=label_name)

        axes[1].plot(np.arange(1, len(best_param_metrics["val_acc"]) + 1, 1), best_param_metrics["val_acc"],
                     markersize=4, label=label_name,  # marker=m,
                     color=c)

        axes[2].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                     markersize=4)

        axes[3].plot(np.arange(1, len(best_param_metrics["train_loss"]) + 1, 1), best_param_metrics["train_loss"],
                     markersize=4, label=label_name,  # marker=m,
                     color=c)
        axes[4].plot(np.arange(1, len(best_param_metrics["val_loss"]) + 1, 1), best_param_metrics["val_loss"],
                     markersize=4, label=label_name,  # marker=m,
                     color=c)

    axes[3].grid(alpha=0.2)
    axes[3].set_title("Training Loss", fontsize=10)
    # axes[3].legend(fontsize=8)
    axes[3].set_yscale('log')

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy", fontsize=10)
    # axes[1].legend(fontsize=8)

    axes[2].grid(alpha=0.2)
    # axes[2].set_title("Validation Acc / Compression Rate", fontsize=10)
    axes[2].set_xlabel("Overall Compression", fontsize=8)
    axes[2].set_ylabel("Test Accuracy", fontsize=8)
    axes[2].legend(fontsize=7)
    axes[2].set_xscale('log')

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Training Accuracy", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[4].grid(alpha=0.2)
    # axes[4].legend(fontsize=8)
    axes[4].set_title("Test Loss", fontsize=10)
    axes[4].set_yscale('log')

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    table = axes[5].table(cellText=table_data,
                          colLabels=["Method", "Test Acc", "Compression Ratio"],
                          cellLoc='center',
                          loc='center')

    axes[5].axis('off')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(9)
            cell._text.set_weight('bold')
        if col == 0 and row > 0:
            cell.set_fontsize(7)

    plt.suptitle(parent_folder.upper())
    plt.tight_layout()
    # plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_compression_metrics("EFSIGNSGD", "baseline_vgg", "training_SGD_OneBitSGD_vgg11_09_01_14_44_24.json")

    plot_compare_all("baseline_vgg", True)

    # plot_compression_rates()
