import ast
import json
import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import itertools

import pandas as pd
from matplotlib import gridspec

from plot_utils import names, markers, colors


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


def plot_compression_metrics(title: str, parent_folder: str):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)
    metrics = {}

    for file_path in all_files:
        if "DS" in file_path:
            continue

        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        if (title in strat["compression"].lower() or title in strat["optimizer"].lower()) or (
                strat["compression"] == "none" and strat["optimizer"] == "sgd"):
            print(strat)

            strat_key = f"{strat}"
            lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

            train_acc = np.array(ast.literal_eval(file["training_acc"]))
            train_loss = np.array(ast.literal_eval(file["training_loss"]))
            val_acc = np.array(ast.literal_eval(file["val_acc"]))
            val_loss = np.array(ast.literal_eval(file["val_loss"]))
            print(len(val_loss))
            cr = file["compression_rates"][0]

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
                                                                                                  strat_key][
                                                                                                  "train_acc"])
                metrics[lean_strat_key][strat_key]["train_loss"] = mean_of_arrays_with_padding(train_loss,
                                                                                               metrics[lean_strat_key][
                                                                                                   strat_key][
                                                                                                   "train_loss"])
                metrics[lean_strat_key][strat_key]["val_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                            metrics[lean_strat_key][
                                                                                                strat_key]["val_acc"])
                metrics[lean_strat_key][strat_key]["val_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                             metrics[lean_strat_key][
                                                                                                 strat_key]["val_loss"])
                metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
                metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                    metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])
    gs = gridspec.GridSpec(3, 2)  # , height_ratios=[1, 1, 1])

    fig = plt.figure(figsize=(11, 12))

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :])  # This subplot spans both columns in the third row
    ]

    # axes = axes.flatten()

    table_data = []
    cs = iter(["g", "r", "y", "purple", "orange", "pink"])

    for method in metrics:
        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)
        axes[4].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], color="black", alpha=0.2)
        for param in metrics[method]:
            met = metrics[method][param]
            param = ast.literal_eval(param)
            param.pop("optimizer")
            param.pop("compression")
            param.pop("learning_rate")

            if param == {}:
                label_name = names[method.replace(" none", "")]
            elif method == "sgd terngrad":
                label_name = "TernGrad"
            else:
                label_name = ', '.join(
                    f"{key}: {value}" for key, value in param.items() if
                    value != "None")

            table_data.append(
                [label_name, round(100 * met["max_val_acc"], 2), round(met["cr"], 1)])

            if label_name.lower() == "sgd":
                m = markers[label_name]
                c = colors[label_name]
            elif len(metrics[method]) == 1:
                m = markers[label_name]
                c = colors[label_name]
            else:
                m = markers[names[title.lower()]]
                c = next(cs)  # colors[names[title.lower()]]

            axes[0].plot(np.arange(1, len(met["train_acc"]) + 1, 1), met["train_acc"],
                         markersize=4, color=c,  # marker=m,
                         label=label_name)

            x_values = np.arange(1, len(met["val_acc"]) + 1, 1)
            y_values = list(moving_average(met["val_acc"], WINDOW_SIZE))
            deviations = np.array(met["val_acc"]) - np.array(y_values)

            # Compute the upper and lower bounds
            upper_bound = y_values + deviations
            lower_bound = y_values - deviations

            # Plot the fill between
            axes[1].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.15)
            axes[1].plot(x_values, y_values, markersize=4, label=label_name, color=c)

            axes[4].plot(met["cr"], met["max_val_acc"], marker=m, label=label_name, color=c,
                         markersize=4)

            axes[3].plot(np.arange(1, len(met["train_loss"]) + 1, 1), met["train_loss"],
                         markersize=4, label=label_name,  # marker=m,
                         color=c)

            x_values = np.arange(1, len(met["val_loss"]) + 1, 1)
            y_values = list(moving_average(met["val_loss"], WINDOW_SIZE))
            deviations = np.array(met["val_loss"]) - np.array(y_values)

            # Compute the upper and lower bounds
            upper_bound = y_values + deviations
            lower_bound = y_values - deviations

            # Plot the fill between
            axes[2].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.15)
            axes[2].plot(x_values, y_values, markersize=4, label=label_name, color=c)

    axes[3].grid(alpha=0.2)
    axes[3].set_title("Training Loss", fontsize=12, fontweight='bold')
    # axes[3].legend(fontsize=8)
    axes[3].set_yscale('log')
    axes[3].tick_params(axis='both', which='major', labelsize=12)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].legend(fontsize=12)

    axes[4].grid(alpha=0.2)
    axes[4].set_title("Test Accuracy vs Overall Compression", fontsize=12, fontweight='bold')
    # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
    # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
    axes[4].legend(fontsize=12)  # , bbox_to_anchor=(0.75, 0.6))
    axes[4].set_xscale('log')
    axes[4].tick_params(axis='both', which='major', labelsize=12)
    axes[4].set_xlim([0.9, 500])
    axes[4].set_ylim([0.9, 1])

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Training Accuracy", fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    axes[2].grid(alpha=0.2)
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].set_title("Test Loss", fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    # axes[4].legend(fontsize=10)

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    # table = axes[5].table(cellText=table_data,
    #                       colLabels=[f"Config", "Test Acc", "Compression Ratio"],
    #                       cellLoc='center',
    #                       loc='center')
    #
    # axes[5].axis('off')
    #
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.scale(1.1, 1.6)
    #
    # for (row, col), cell in table.get_celld().items():
    #     if row == 0:
    #         cell.set_fontsize(11)
    #         cell._text.set_weight('bold')
    #     if col == 0 and row > 0:
    #         cell.set_fontsize(11)
    # table.auto_set_column_width(col=list(range(3)))

    # plt.suptitle(names[title.lower()], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../../figures/" + parent_folder + "_" + title + ".pdf", bbox_inches='tight')
    # plt.show()


def moving_average(data, window_size):
    cumsum = [0]
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= window_size:
            moving_ave = (cumsum[i] - cumsum[i - window_size]) / window_size
            yield moving_ave
        else:
            yield cumsum[i] / i


def moving_std(data, window_size):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    moving_avg = ret[window_size - 1:] / window_size
    moving_var = (np.cumsum(np.power(data, 2), dtype=float)[window_size - 1:] - np.power(moving_avg,
                                                                                         2) * window_size) / (
                         window_size - 1)
    return np.sqrt(moving_var)


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
    return np.mean([arr1_padded, arr2_padded], axis=0)

    return mean_arr


def plot_compare_all(parent_folder: str, bsgd: bool, epochs: int, save=False):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        if ("Bucket" in file_path or "SGD_mom" in file_path) and not bsgd:
            continue
        # if not "MEM" in file_path:
        #     continue
        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        train_acc = np.array(ast.literal_eval(file["training_acc"]))[:epochs]
        train_loss = np.array(ast.literal_eval(file["training_loss"]))[:epochs]
        val_acc = np.array(ast.literal_eval(file["val_acc"]))[:epochs]
        val_loss = np.array(ast.literal_eval(file["val_loss"]))[:epochs]
        cr = file["compression_rates"][0]

        print(len(train_loss))
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
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
        m = markers[label_name]
        c = colors[label_name]

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        axes[0].plot(np.arange(1, min(len(best_param_metrics["train_acc"]) + 1, epochs + 1), 1),
                     best_param_metrics["train_acc"][:epochs],
                     markersize=4, color=c,
                     label=label_name)

        x_values = np.arange(1, min(len(best_param_metrics["val_acc"]) + 1, epochs + 1), 1)
        y_values = list(moving_average(best_param_metrics["val_acc"], WINDOW_SIZE))[:epochs]
        deviations = np.array(best_param_metrics["val_acc"])[:epochs] - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[1].fill_between(x_values, lower_bound[:epochs], upper_bound[:epochs], color=c, alpha=0.3)
        axes[1].plot(x_values, y_values, markersize=4, label=label_name, color=c)

        axes[2].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                     markersize=4)

        axes[3].plot(np.arange(1, min(len(best_param_metrics["train_loss"]) + 1, epochs + 1), 1),
                     best_param_metrics["train_loss"][:epochs],
                     markersize=4, label=label_name,  # marker=m,
                     color=c)

        x_values = np.arange(1, min(len(best_param_metrics["val_loss"]) + 1, epochs + 1), 1)
        y_values = list(moving_average(best_param_metrics["val_loss"], WINDOW_SIZE))[:epochs]
        deviations = np.array(best_param_metrics["val_loss"])[:epochs] - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[4].fill_between(x_values, lower_bound[:epochs], upper_bound[:epochs], color=c, alpha=0.3)
        axes[4].plot(x_values, y_values, markersize=4, label=label_name, color=c)

    axes[3].grid(alpha=0.2)
    axes[3].set_title("Training Loss", fontsize=10, fontweight='bold')
    # axes[3].legend(fontsize=8)
    axes[3].set_yscale('log')
    axes[3].tick_params(axis='both', which='major', labelsize=8)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy", fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=8)

    axes[2].grid(alpha=0.2)
    axes[2].set_title("Test Accuracy vs Overall Compression", fontsize=10, fontweight='bold')
    # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
    # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
    axes[2].legend(fontsize=7)  # , bbox_to_anchor=(0.75, 0.7))
    axes[2].set_xscale('log')
    axes[2].tick_params(axis='both', which='major', labelsize=8)

    if "lenet_batchdescent" in parent_folder:
        axes[2].set_xlim([0.9, 500])
        axes[2].set_ylim([0.85, 1])
    elif "lenet" in parent_folder:
        axes[2].set_xlim([0.9, 500])
        axes[2].set_ylim([0.93, 1])
    else:
        axes[2].set_xlim([0.9, 500])
        axes[2].set_ylim([0.6, 0.8])

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Training Accuracy", fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='both', which='major', labelsize=8)

    axes[4].grid(alpha=0.2)
    axes[4].tick_params(axis='both', which='major', labelsize=8)
    axes[4].set_title("Test Loss", fontsize=10, fontweight='bold')
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
            cell.set_fontsize(8)
    table.auto_set_column_width(col=list(range(3)))

    if not save:
        plt.suptitle(parent_folder.upper())
    plt.tight_layout()
    if save:
        plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


def plot_compare_all_selected(selected_method, parent_folder: str, bsgd: bool, epochs: int, save=False,
                              selected=None):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        if ("Bucket" in file_path or "SGD_mom" in file_path) and not bsgd:
            continue

        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        train_acc = np.array(ast.literal_eval(file["training_acc"]))[:epochs]
        train_loss = np.array(ast.literal_eval(file["training_loss"]))[:epochs]
        val_acc = np.array(ast.literal_eval(file["val_acc"]))[:epochs]
        val_loss = np.array(ast.literal_eval(file["val_loss"]))[:epochs]
        cr = file["compression_rates"][0]

        print(len(train_loss))
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

    if selected is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    table_data = []
    for method in metrics:
        label_name = names[method.lower().replace(" none", "")]
        if method.lower().replace(" none", "") not in selected_method and len(selected_method) > 0:
            print("cont", method.lower().replace(" none", ""))
            continue
        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))
        best_param = max(metrics[method].items(), key=lambda x: x[1]['max_val_acc'])
        best_param_metrics = best_param[1]

        m = markers[label_name]
        c = colors[label_name]

        table_data.append(
            [label_name, round(100 * best_param_metrics["max_val_acc"], 2), round(best_param_metrics["cr"], 1)])
        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        if selected == 0:
            axes.plot(np.arange(1, min(len(best_param_metrics["train_acc"]) + 1, epochs + 1), 1),
                      best_param_metrics["train_acc"][:epochs],
                      markersize=4, color=c,
                      label=label_name)

        x_values = np.arange(1, min(len(best_param_metrics["val_acc"]) + 1, epochs + 1), 1)
        y_values = list(moving_average(best_param_metrics["val_acc"], WINDOW_SIZE))[:epochs]
        deviations = np.array(best_param_metrics["val_acc"])[:epochs] - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        if selected == 1:
            axes.fill_between(x_values, lower_bound[:epochs], upper_bound[:epochs], color=c, alpha=0.3)
            axes.plot(x_values, y_values, markersize=4, label=label_name, color=c)
        if selected == 2:
            axes.plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                      markersize=4)
        if selected == 3:
            axes.plot(np.arange(1, min(len(best_param_metrics["train_loss"]) + 1, epochs + 1), 1),
                      best_param_metrics["train_loss"][:epochs],
                      markersize=4, label=label_name,  # marker=m,
                      color=c)

        x_values = np.arange(1, min(len(best_param_metrics["val_loss"]) + 1, epochs + 1), 1)
        y_values = list(moving_average(best_param_metrics["val_loss"], WINDOW_SIZE))[:epochs]
        deviations = np.array(best_param_metrics["val_loss"])[:epochs] - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        if selected == 4:
            axes.fill_between(x_values, lower_bound[:epochs], upper_bound[:epochs], color=c, alpha=0.3)
            axes.plot(x_values, y_values, markersize=4, label=label_name, color=c)

    if selected == 3:
        axes.grid(alpha=0.2)
        axes.set_title("Training Loss", fontsize=12, fontweight='bold')
        # axes[3].legend(fontsize=8)
        axes.set_yscale('log')
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.set_xlabel("Epochs", fontsize=12)

    if selected == 1:
        axes.grid(alpha=0.2)
        axes.set_title("Test Accuracy", fontsize=12, fontweight='bold')
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.legend(fontsize=12)
        axes.set_xlabel("Epochs", fontsize=12)
        axes.set_ylim([0.4, 0.8])

    if selected == 2:
        axes.grid(alpha=0.2)
        axes.set_title("Test Accuracy vs Overall Compression", fontsize=12, fontweight='bold')
        # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
        # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
        axes.legend(fontsize=12)  # , bbox_to_anchor=(0.75, 0.7))
        axes.set_xscale('log')
        axes.tick_params(axis='both', which='major', labelsize=12)
        if "lenet" in parent_folder:
            axes.set_xlim([0.9, 500])
            axes.set_ylim([0.82, 1])
        else:
            axes.set_xlim([0.9, 1000])
            axes.set_ylim([0.5, 0.78])
    if selected == 0:
        axes.grid(alpha=0.2)
        axes.set_title("Training Accuracy", fontsize=12, fontweight='bold')
        axes.legend(fontsize=12)
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.set_xlabel("Epochs", fontsize=12)

    if selected == 4:
        axes.grid(alpha=0.2)
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.set_title("Test Loss", fontsize=12, fontweight='bold')
        axes.set_yscale('log')
        axes.legend(fontsize=12)
        axes.set_xlabel("Epochs", fontsize=12)

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)
    if selected == 5:
        table = axes.table(cellText=table_data,
                           colLabels=["Method", "Test Acc", "Compression Ratio"],
                           cellLoc='center',
                           loc='center')

        axes.axis('off')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.6)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_fontsize(9)
                cell._text.set_weight('bold')
            if col == 0 and row > 0:
                cell.set_fontsize(8)
        table.auto_set_column_width(col=list(range(3)))

    if not save:
        plt.suptitle(parent_folder.upper())
    plt.tight_layout()
    if save:
        plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


def plot_bit_all(parent_folder: str, bsgd: bool, max_bits_ratio, epochs):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    bits_pro_epoch = 1000

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        if ("Bucket" in file_path or "SGD_mom" in file_path) and not bsgd:
            continue
        # if not "MEM" in file_path:
        #     continue
        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        cr = file["compression_rates"][0]
        if False:  # cr == 1:
            continue
            train_acc = np.array(ast.literal_eval(file["training_acc"]))
            train_loss = np.array(ast.literal_eval(file["training_loss"]))
            val_acc = np.array(ast.literal_eval(file["val_acc"]))
            val_loss = np.array(ast.literal_eval(file["val_loss"]))
        else:
            eps = len(np.array(ast.literal_eval(file["training_acc"])))
            print(bits_pro_epoch / cr)
            cut_off = int((eps * bits_pro_epoch * max_bits_ratio) / (eps * (bits_pro_epoch / cr)) * eps)
            print(lean_strat_key, cut_off, eps)
            train_acc = np.array(ast.literal_eval(file["training_acc"]))[:epochs][:cut_off]
            train_loss = np.array(ast.literal_eval(file["training_loss"]))[:epochs][:cut_off]
            val_acc = np.array(ast.literal_eval(file["val_acc"]))[:epochs][:cut_off]
            val_loss = np.array(ast.literal_eval(file["val_loss"]))[:epochs][:cut_off]

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
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
        m = markers[label_name]
        c = colors[label_name]

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        current_timeline = np.linspace(0, 100, len(best_param_metrics["train_acc"]))

        # Create the fixed "bits" timeline
        fixed_timeline = np.linspace(0, 100, 100)

        # Interpolate the data onto the fixed timeline
        interpolated_acc = np.interp(fixed_timeline, current_timeline, best_param_metrics["train_acc"])

        axes[0].plot(interpolated_acc,
                     markersize=4, color=c,
                     label=label_name)

        interpolated_test_acc = np.interp(fixed_timeline, current_timeline, best_param_metrics["val_acc"])
        x_values = np.arange(len(interpolated_test_acc))
        y_values = list(moving_average(interpolated_test_acc, WINDOW_SIZE))
        deviations = np.array(interpolated_test_acc) - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[1].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.3)
        axes[1].plot(y_values, markersize=4, label=label_name, color=c)

        axes[2].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                     markersize=4)

        interpolated_train_loss = np.interp(fixed_timeline, current_timeline, best_param_metrics["train_loss"])
        axes[3].plot(interpolated_train_loss,
                     markersize=4, label=label_name,
                     color=c)

        interpolated_test_loss = np.interp(fixed_timeline, current_timeline, best_param_metrics["val_loss"])
        x_values = np.arange(len(interpolated_test_loss))
        y_values = list(moving_average(interpolated_test_loss, WINDOW_SIZE))
        deviations = np.array(interpolated_test_loss) - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[4].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.3)
        axes[4].plot(y_values,
                     markersize=4, label=label_name,
                     color=c)

    axes[3].grid(alpha=0.2)
    axes[3].set_title("Training Loss", fontsize=10, fontweight='bold')
    # axes[3].legend(fontsize=8)
    axes[3].set_yscale('log')
    axes[3].tick_params(axis='both', which='major', labelsize=8)
    axes[3].set_xlabel("% available bits used", fontsize=8)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy", fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=8)
    axes[1].set_xlabel("% available bits used", fontsize=8)

    axes[2].grid(alpha=0.2)
    axes[2].set_title("Test Accuracy vs Overall Compression", fontsize=10, fontweight='bold')
    axes[2].legend(fontsize=7)  # , bbox_to_anchor=(0.75, 0.7))
    axes[2].set_xscale('log')
    axes[2].tick_params(axis='both', which='major', labelsize=8)
    axes[2].set_xlim([0.9, 500])
    axes[2].set_ylim([0.93, 1])

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Training Accuracy", fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='both', which='major', labelsize=8)
    axes[0].set_xlabel("% available bits used", fontsize=8)
    axes[4].grid(alpha=0.2)
    axes[4].tick_params(axis='both', which='major', labelsize=8)
    axes[4].set_title("Test Loss", fontsize=10, fontweight='bold')
    axes[4].set_yscale('log')
    axes[4].set_xlabel("% available bits used", fontsize=8)

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
            cell.set_fontsize(8)
    table.auto_set_column_width(col=list(range(3)))

    # plt.suptitle("Bits timeline " + parent_folder.upper())
    plt.tight_layout()
    plt.savefig(f"../../figures/{parent_folder}_{max_bits_ratio}_used.pdf", bbox_inches='tight')
    plt.show()


def plot_time_all(parent_folder: str, bsgd: bool):
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
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        cr = file["compression_rates"][0]

        train_acc = np.array(ast.literal_eval(file["training_acc"]))
        train_loss = np.array(ast.literal_eval(file["training_loss"]))
        val_acc = np.array(ast.literal_eval(file["val_acc"]))
        val_loss = np.array(ast.literal_eval(file["val_loss"]))
        time_per_epoch = np.array(file["time_per_epoch"])

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
            metrics[lean_strat_key][strat_key]["time_per_epoch"] = time_per_epoch

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

            metrics[lean_strat_key][strat_key]["time_per_epoch"] = mean_of_arrays_with_padding(time_per_epoch,
                                                                                               metrics[lean_strat_key][
                                                                                                   strat_key][
                                                                                                   "time_per_epoch"])

            metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
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
        m = markers[label_name]
        c = colors[label_name]

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        WINDOW_SIZE = 2

        cumulative_time = np.cumsum(best_param_metrics["time_per_epoch"])

        axes[0].plot(cumulative_time,
                     best_param_metrics["train_acc"],
                     markersize=4, color=c,
                     label=label_name)

        x_values = np.arange(1, len(best_param_metrics["val_acc"]) + 1, 1)
        y_values = list(moving_average(best_param_metrics["val_acc"], WINDOW_SIZE))
        deviations = np.array(best_param_metrics["val_acc"]) - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        # axes[1].fill_between(x_values, lower_bound[:epochs], upper_bound[:epochs], color=c, alpha=0.3)
        axes[1].plot(cumulative_time, y_values, markersize=4, label=label_name, color=c)

        axes[2].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                     markersize=4)

        axes[3].plot(cumulative_time,
                     best_param_metrics["train_loss"],
                     markersize=4, label=label_name,  # marker=m,
                     color=c)
        axes[4].plot(cumulative_time,
                     list(moving_average(best_param_metrics["val_loss"], WINDOW_SIZE)),
                     markersize=4, label=label_name,  # marker=m,
                     color=c)

    axes[3].grid(alpha=0.2)
    axes[3].set_title("Training Loss", fontsize=10, fontweight='bold')
    axes[3].set_xlabel("seconds", fontsize=8)
    axes[3].set_yscale('log')
    axes[3].tick_params(axis='both', which='major', labelsize=8)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy", fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=8)
    axes[1].set_xlabel("seconds", fontsize=8)

    axes[2].grid(alpha=0.2)
    axes[2].set_title("Test Accuracy vs Overall Compression", fontsize=10, fontweight='bold')
    # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
    # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
    axes[2].legend(fontsize=7)  # , bbox_to_anchor=(0.75, 0.7))
    axes[2].set_xscale('log')
    axes[2].tick_params(axis='both', which='major', labelsize=8)
    axes[2].set_xlim([0.9, 500])
    axes[2].set_ylim([0.93, 1])

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Training Accuracy", fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='both', which='major', labelsize=8)
    axes[0].set_xlabel("seconds", fontsize=8)

    axes[4].grid(alpha=0.2)
    axes[4].tick_params(axis='both', which='major', labelsize=8)
    axes[4].set_title("Test Loss", fontsize=10, fontweight='bold')
    axes[4].set_yscale('log')
    axes[4].set_xlabel("seconds", fontsize=8)
    axes[4].set_xscale('log')

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
            cell.set_fontsize(8)
    table.auto_set_column_width(col=list(range(3)))

    plt.suptitle(parent_folder.upper())
    plt.tight_layout()
    # plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


def plot_bit_all2(parent_folder: str, bsgd: bool):
    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    bits_pro_epoch = 61706 * 32
    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        if ("Bucket" in file_path or "SGD_mom" in file_path) and not bsgd:
            continue
        # if not "MEM" in file_path:
        #     continue
        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        cr = file["compression_rates"][0]

        train_acc = np.array(ast.literal_eval(file["training_acc"]))
        train_loss = np.array(ast.literal_eval(file["training_loss"]))
        val_acc = np.array(ast.literal_eval(file["val_acc"]))
        val_loss = np.array(ast.literal_eval(file["val_loss"]))
        time_per_epoch = np.array(file["time_per_epoch"])

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

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    table_data = []
    for method in metrics:
        label_name = names[method.lower().replace(" none", "")]

        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))
        best_param = max(metrics[method].items(), key=lambda x: x[1]['max_val_acc'])
        best_param_metrics = best_param[1]

        m = markers[label_name]
        c = colors[label_name]

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)

        eps = len(best_param_metrics["train_acc"])
        sum_bits = eps * (bits_pro_epoch / best_param_metrics["cr"])

        table_data.append(
            [label_name, round(sum_bits / 8e+6, 2), round(100 * (eps * bits_pro_epoch) / sum_bits, 1)])

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    table_data = [[t[0], str(t[1]) + "MB", str(t[2]) + "%"] for t in table_data]
    table_data[0][-1] = "-"

    table = axes.table(cellText=table_data,
                       colLabels=["Method", "Total Data Communication", "Communication Cost Saving"],
                       cellLoc='center',
                       loc='center')

    axes.axis('off')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(10)
            cell._text.set_weight('bold')
        if col == 0 and row > 0:
            cell.set_fontsize(9)
    table.auto_set_column_width(col=list(range(3)))

    plt.tight_layout()
    plt.savefig(f"../../figures/{parent_folder}_total_bit_sent.pdf", bbox_inches='tight')
    plt.show()


def plot_compare_batches(parent_folders: list, epochs):
    all_files = []
    for parent_folder in parent_folders:
        directory_path = '../results/compression/' + parent_folder
        all_files += get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}" + "&{}".format(file["setup"]["batch_size"])
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        train_acc = np.array(ast.literal_eval(file["training_acc"]))[:epochs]
        train_loss = np.array(ast.literal_eval(file["training_loss"]))[:epochs]
        val_acc = np.array(ast.literal_eval(file["val_acc"]))[:epochs]
        val_loss = np.array(ast.literal_eval(file["val_loss"]))[:epochs]
        cr = file["compression_rates"][0]
        bs = file["setup"]["batch_size"]
        print(len(train_loss))
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
            metrics[lean_strat_key][strat_key]["batch_size"] = bs

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
            metrics[lean_strat_key][strat_key]["batch_size"] = bs

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    vgl = {}
    table_data = []
    for method in metrics:
        cr_acc_pairs = []
        for p in metrics[method]:
            if p.split("&")[-1] == "64":
                continue
            cr_acc_pairs.append(metrics[method][p])
        best_param = max(cr_acc_pairs, key=lambda x: x['max_val_acc'])
        best_param_metrics_32 = best_param

        cr_acc_pairs = []
        for p in metrics[method]:
            if p.split("&")[-1] == "32":
                continue
            cr_acc_pairs.append(metrics[method][p])
        best_param = max(cr_acc_pairs, key=lambda x: x['max_val_acc'])
        best_param_metrics_64 = best_param

        print(best_param_metrics_32)
        print(best_param_metrics_64)

        label_name = names[method.replace("none", "")]
        print(label_name)

        vgl[label_name] = {
            "32": best_param_metrics_32,
            "64": best_param_metrics_64
        }

        ax.axis("off")

        table_data.append(
            [label_name, round(100 * (best_param_metrics_32['max_val_acc'] - best_param_metrics_64['max_val_acc']), 2)])

        # table_data = sorted(table_data, key=lambda x: x[1], reverse=True)
        #
        # table_data = [[t[0], str(t[1]) + "MB", str(t[2]) + "%"] for t in table_data]
        # table_data[0][-1] = "-"

    table = ax.table(cellText=table_data,
                     colLabels=["Method", "Total Data Communication"],
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    table.auto_set_column_width(col=list(range(2)))

    plt.show()


if __name__ == "__main__":
    WINDOW_SIZE = 3

    # plot_compression_metrics("topk", "lenet_64")

    # plot_compare_all("lenet_batchdescent", True, 500, save=True)

    plot_compare_all_selected(
        ["sgd", "efsignsgd", "sgd onebitsgd", "sgd terngrad", "sgd naturalcompression"],
        "resnet_500",
        True, 500, save=True, selected=1)

    # plot_compression_rates()

    # plot_bit_all("lenet_64", False, 0.05, 60)

    # plot_bit_all2("lenet_32", False)

    # plot_time_all("lenet_32", True)

    # plot_compare_batches(["lenet_32", "lenet_64"], 60)
