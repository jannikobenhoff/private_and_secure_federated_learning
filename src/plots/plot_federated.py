import ast
import json
import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import itertools

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
from skopt.learning.gaussian_process.kernels import Matern

from plot_utils import names, colors, markers


def get_all_files_in_directory(root_path):
    all_files = []
    for subdir, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            all_files.append(file_path)
    return all_files


def plot_compression_metrics(title: str, parent_folders: list, save=False):
    all_files = []
    for parent_folder in parent_folders:
        directory_path = '../results/federated/' + parent_folder
        all_files += get_all_files_in_directory(directory_path)
    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue

        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        if (title in strat["compression"].lower() or title in strat["optimizer"].lower()) or (
                strat["compression"] == "none" and strat["optimizer"] == "sgd"):

            strat_key = f"{strat} & {file['args']['local_iter_type']} {file['args']['beta']}"
            lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

            val_acc = np.array(ast.literal_eval(file["test_acc"]))
            val_loss = np.array(ast.literal_eval(file["test_loss"]))

            cr = file["compression_rates"][0]

            if lean_strat_key not in metrics:
                metrics[lean_strat_key] = {}

            if strat_key not in metrics[lean_strat_key]:
                metrics[lean_strat_key][strat_key] = {}
                metrics[lean_strat_key][strat_key]["test_acc"] = val_acc
                metrics[lean_strat_key][strat_key]["test_loss"] = val_loss
                metrics[lean_strat_key][strat_key]["cr"] = cr
                metrics[lean_strat_key][strat_key]["max_val_acc"] = np.max(val_acc)

            else:
                metrics[lean_strat_key][strat_key]["test_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                             metrics[lean_strat_key][
                                                                                                 strat_key]["test_acc"])
                metrics[lean_strat_key][strat_key]["test_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                              metrics[lean_strat_key][
                                                                                                  strat_key][
                                                                                                  "test_loss"])
                metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
                metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                    metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    marker = itertools.cycle(('+', 'v', 'o', '*'))
    cs = iter(["b", "g", "r", "y", "b", "g", "r", "y"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 1.3]})
    axes = axes.flatten()

    table_data = []
    for method in metrics:
        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))

        sorted_data = sorted(cr_acc_pairs, key=lambda x: x[0], reverse=True)
        axes[2].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], color="black", alpha=0.2)
        for param in metrics[method]:
            met = metrics[method][param]
            setup = param.split("&")[-1]
            param = ast.literal_eval(param.split("&")[0])
            param.pop("optimizer")
            param.pop("compression")
            if param == {}:
                label_name = names[method.replace(" none", "")] + setup
            else:
                label_name = ', '.join(
                    f"{key}: {value}" for key, value in param.items() if
                    value != "None") + setup

            table_data.append(
                [label_name, round(100 * met["max_val_acc"], 2), round(met["cr"], 1)])
            m = next(marker)
            # c = colors[label_name]
            c = next(cs)
            axes[0].plot(np.arange(1, len(met["test_acc"]) + 1, 1), list(moving_average(met["test_acc"], WINDOW_SIZE)),
                         markersize=4, label=label_name,  # marker=m,
                         color=c)

            axes[1].plot(met["cr"], met["max_val_acc"], marker=m, label=label_name, color=c,
                         markersize=4)

            axes[2].plot(np.arange(1, len(met["test_loss"]) + 1, 1),
                         list(moving_average(met["test_loss"], WINDOW_SIZE)),
                         markersize=4, label=label_name,  # marker=m,
                         color=c)

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Test Accuracy", fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=8)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy vs Overall Compression", fontsize=10, fontweight='bold')
    # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
    # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
    axes[1].legend(fontsize=7)  # , bbox_to_anchor=(0.75, 0.6))
    axes[1].set_xscale('log')
    axes[1].tick_params(axis='both', which='major', labelsize=8)

    axes[2].grid(alpha=0.2)
    axes[2].tick_params(axis='both', which='major', labelsize=8)
    axes[2].set_title("Test Loss", fontsize=10, fontweight='bold')
    axes[2].set_yscale('log')

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    table = axes[3].table(cellText=table_data,
                          colLabels=[f"Config", "Test Acc", "Compression Ratio"],
                          cellLoc='center',
                          loc='center')

    axes[3].axis('off')

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
        plt.suptitle(names[title.lower()], fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("../../figures/" + title + "_" + parent_folder + ".pdf", bbox_inches='tight')
    plt.show()


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


def plot_compare_all(parent_folder: str, limit):
    directory_path = '../results/federated/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        file = open(file_path, "r")
        file = json.load(file)
        print(file["args"]["local_iter_type"], file["args"]["beta"])
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}"  # {file['args']['lambda_l2']}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        # train_acc = np.array(ast.literal_eval(file["training_acc"]))
        # train_loss = np.array(ast.literal_eval(file["training_loss"]))
        val_acc = np.array(ast.literal_eval(file["test_acc"]))
        val_loss = np.array(ast.literal_eval(file["test_loss"]))
        cr = file["compression_rates"][0]

        # print(strat_key, np.max(val_acc))
        if lean_strat_key not in metrics:
            metrics[lean_strat_key] = {}

        if strat_key not in metrics[lean_strat_key]:
            metrics[lean_strat_key][strat_key] = {}
            # metrics[lean_strat_key][strat_key]["train_acc"] = train_acc
            # metrics[lean_strat_key][strat_key]["train_loss"] = train_loss
            metrics[lean_strat_key][strat_key]["test_acc"] = val_acc
            metrics[lean_strat_key][strat_key]["test_loss"] = val_loss
            metrics[lean_strat_key][strat_key]["cr"] = cr
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.max(val_acc)

        else:
            # metrics[lean_strat_key][strat_key]["train_acc"] = mean_of_arrays_with_padding(train_acc,
            #                                                                               metrics[lean_strat_key][
            #                                                                                   strat_key]["train_acc"])
            # metrics[lean_strat_key][strat_key]["train_loss"] = mean_of_arrays_with_padding(train_loss,
            #                                                                                metrics[lean_strat_key][
            #                                                                                    strat_key]["train_loss"])
            metrics[lean_strat_key][strat_key]["test_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                         metrics[lean_strat_key][
                                                                                             strat_key]["test_acc"])
            metrics[lean_strat_key][strat_key]["test_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                          metrics[lean_strat_key][
                                                                                              strat_key]["test_loss"])
            metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1.15]})
    axes = axes.flatten()

    table_data = []
    for method in metrics:
        label_name = names[method.lower().replace(" none", "").replace("none ", "")]

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

        x_values = np.arange(1, len(best_param_metrics["test_acc"]) + 1, 1)
        y_values = list(moving_average(best_param_metrics["test_acc"], WINDOW_SIZE))
        deviations = np.array(best_param_metrics["test_acc"]) - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[0].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.3)
        axes[0].plot(x_values, y_values, markersize=4, label=label_name, color=c)

        axes[1].plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], marker=m, label=label_name, color=c,
                     markersize=4)

        x_values = np.arange(1, len(best_param_metrics["test_loss"]) + 1, 1)
        y_values = list(moving_average(best_param_metrics["test_loss"], WINDOW_SIZE))
        deviations = np.array(best_param_metrics["test_loss"]) - np.array(y_values)

        # Compute the upper and lower bounds
        upper_bound = y_values + deviations
        lower_bound = y_values - deviations

        # Plot the fill between
        axes[2].fill_between(x_values, lower_bound, upper_bound, color=c, alpha=0.3)
        axes[2].plot(x_values, y_values, markersize=4, label=label_name, color=c)

    axes[0].grid(alpha=0.2)
    axes[0].set_title("Test Accuracy", fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=8)
    axes[0].legend(fontsize=7)

    axes[1].grid(alpha=0.2)
    axes[1].set_title("Test Accuracy vs Overall Compression", fontsize=10, fontweight='bold')
    # axes[2].set_xlabel("Overall Compression", fontsize=8, fontweight='bold')
    # axes[2].set_ylabel("Test Accuracy", fontsize=8, fontweight='bold')
    axes[1].legend(fontsize=7)  # , bbox_to_anchor=(0.75, 0.7))
    axes[1].set_xscale('log')
    axes[1].tick_params(axis='both', which='major', labelsize=8)
    axes[1].set_ylim(limit)

    axes[2].grid(alpha=0.2)
    axes[2].tick_params(axis='both', which='major', labelsize=8)
    axes[2].set_title("Test Loss", fontsize=10, fontweight='bold')
    axes[2].set_xlabel("Iteration", fontsize=8)
    axes[2].set_yscale('log')

    table_data = sorted(table_data, key=lambda x: x[1], reverse=True)

    table = axes[3].table(cellText=table_data,
                          colLabels=["Method", "Test Acc", "Compression Ratio"],
                          cellLoc='center',
                          loc='center')

    axes[3].axis('off')

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
    plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


def lambda_search(parent_folder):
    directory_path = '../results/federated/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        file = open(file_path, "r")
        file = json.load(file)
        if file["args"]["local_iter_type"] == "same":
            continue
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat} {file['args']['lambda_l2']}"
        lean_strat_key = f"{strat['optimizer']} {strat['compression']} {file['args']['lambda_l2']}"

        val_acc = np.array(ast.literal_eval(file["test_acc"]))
        val_loss = np.array(ast.literal_eval(file["test_loss"]))
        l2_lambda = file['args']['lambda_l2']
        cr = file["compression_rates"][0]

        if lean_strat_key not in metrics:
            metrics[lean_strat_key] = {}

        if strat_key not in metrics[lean_strat_key]:
            metrics[lean_strat_key][strat_key] = {}
            metrics[lean_strat_key][strat_key]["test_acc"] = val_acc
            metrics[lean_strat_key][strat_key]["test_loss"] = val_loss
            metrics[lean_strat_key][strat_key]["cr"] = cr
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.max(val_acc)
            metrics[lean_strat_key][strat_key]["l2_lambda"] = l2_lambda

        else:
            metrics[lean_strat_key][strat_key]["test_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                         metrics[lean_strat_key][
                                                                                             strat_key]["test_acc"])
            metrics[lean_strat_key][strat_key]["test_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                          metrics[lean_strat_key][
                                                                                              strat_key]["test_loss"])
            metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))  # , gridspec_kw={'width_ratios': [1, 1.3]})
    axes = axes.flatten()
    data_points = [[], []]
    for method in metrics:
        # label_name = names[method.lower().replace(" none", "").replace("none ", "")]
        cr_acc_pairs = []
        for p in metrics[method]:
            cr_acc_pairs.append((metrics[method][p]['cr'], metrics[method][p]['max_val_acc']))
        best_param = max(metrics[method].items(), key=lambda x: x[1]['max_val_acc'])
        best_param_metrics = best_param[1]
        print(best_param_metrics["l2_lambda"], best_param_metrics["max_val_acc"])

        data_points[0].append(best_param_metrics["l2_lambda"])
        data_points[1].append(best_param_metrics["max_val_acc"])
        axes[0].scatter(best_param_metrics["l2_lambda"], best_param_metrics["max_val_acc"])
        axes[1].plot(best_param_metrics["test_acc"], label=best_param_metrics["l2_lambda"])
        axes[2].plot(best_param_metrics["test_loss"])

    axes[0].set_xscale("log")
    axes[0].set_xlim([1e-6, 0.1])
    axes[1].legend()
    plt.show()


def plot_compare_to_diff_sets(parent_folders: list):
    all_files = []
    for parent_folder in parent_folders:
        directory_path = '../results/federated/' + parent_folder
        all_files += get_all_files_in_directory(directory_path)

    metrics = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue
        file = open(file_path, "r")
        file = json.load(file)
        print(file["args"]["local_iter_type"], file["args"]["beta"])
        strat = ast.literal_eval(file["args"]["strategy"])
        strat_key = f"{strat}" + "&{}_{}".format(file["args"]["local_iter_type"], file["args"]["beta"])
        lean_strat_key = f"{strat['optimizer']} {strat['compression']}"

        val_acc = np.array(ast.literal_eval(file["test_acc"]))
        val_loss = np.array(ast.literal_eval(file["test_loss"]))
        cr = file["compression_rates"][0]

        if lean_strat_key not in metrics:
            metrics[lean_strat_key] = {}

        if strat_key not in metrics[lean_strat_key]:
            metrics[lean_strat_key][strat_key] = {}
            metrics[lean_strat_key][strat_key]["test_acc"] = val_acc
            metrics[lean_strat_key][strat_key]["test_loss"] = val_loss
            metrics[lean_strat_key][strat_key]["cr"] = cr
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.max(val_acc)
        else:
            metrics[lean_strat_key][strat_key]["test_acc"] = mean_of_arrays_with_padding(val_acc,
                                                                                         metrics[lean_strat_key][
                                                                                             strat_key]["test_acc"])
            metrics[lean_strat_key][strat_key]["test_loss"] = mean_of_arrays_with_padding(val_loss,
                                                                                          metrics[lean_strat_key][
                                                                                              strat_key]["test_loss"])
            metrics[lean_strat_key][strat_key]["cr"] = np.mean([cr, metrics[lean_strat_key][strat_key]["cr"]])
            metrics[lean_strat_key][strat_key]["max_val_acc"] = np.mean([
                metrics[lean_strat_key][strat_key]["max_val_acc"], np.max(val_acc)])

    fig, axes = plt.subplots(4, 3, figsize=(17, 8))
    axes = axes.flatten()

    ax_index = 0
    for method in metrics:
        print(method)
        label_name = names[method.lower().replace(" none", "").replace("none ", "")]

        setup_data = {"same": {}, "dirichlet": {}}
        for setup in metrics[method]:
            s = setup.split("&")[-1].split("_")
            setup_data[s[0]][s[1]] = metrics[method][setup]["max_val_acc"]

        print(setup_data)
        diff_matrix = np.zeros((4, 4))

        # Getting the keys and values from your data dictionary
        keys = list(setup_data.keys())
        subkeys = list(setup_data[keys[0]].keys())
        if len(keys) + len(subkeys) != 4:
            print("BREAK")
            continue
        # Calculating the differences between each pair of setups
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(subkeys):
                for k, key3 in enumerate(keys):
                    for l, key4 in enumerate(subkeys):
                        diff_matrix[i * len(subkeys) + j][k * len(subkeys) + l] = round(100 * (setup_data[key1][key2] - \
                                                                                               setup_data[key3][key4]),
                                                                                        2)

        # Creating a list of labels for the table
        labels = [f'{key1}{key2}' for key1 in keys for key2 in subkeys]

        # Creating a pandas DataFrame from the differences matrix
        df = pd.DataFrame(diff_matrix, columns=labels, index=labels)

        # Hide axes
        axes[ax_index].xaxis.set_visible(False)
        axes[ax_index].yaxis.set_visible(False)
        axes[ax_index].axis('off')
        axes[ax_index].set_title(label_name)
        # Table from DataFrame
        table_data = []
        for row in df.index:
            table_data.append(df.loc[row])
        table = axes[ax_index].table(cellText=table_data, colLabels=df.columns, loc='center', cellLoc='center',
                                     colColours=['#f2f2f2'] * df.shape[1])

        # Apply color map to the cells
        # norm = plt.cm.colors.Normalize(vmax=abs(diff_matrix).max(), vmin=-abs(diff_matrix).max())
        norm = plt.cm.colors.Normalize(vmax=10, vmin=-10)
        colors = plt.cm.RdBu_r(norm(diff_matrix))
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                table[(i + 1, j)].set_facecolor(colors[i, j])
                table[(i + 1, j)].set_text_props(
                    color="w" if abs(diff_matrix[i, j]) > 0.5 * 10 else "k")

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                # cell.set_fontsize(5)
                cell._text.set_weight('bold')
            if col == 0 and row > 0:
                cell.set_fontsize(8)
        table.auto_set_column_width(col=list(range(4)))

        ax_index += 1
        print("i:", ax_index)

    plt.show()

    # plt.suptitle(parent_folder.upper())
    # plt.tight_layout()
    # plt.savefig(f"../../figures/{parent_folder}.pdf", bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    WINDOW_SIZE = 3
    # plot_compression_metrics("gradientsparsification", ["lenet_same2", "lenet_dirichlet2"], save=False)

    plot_compare_all("lenet_same2", [0.73, 1])

    # plot_compare_to_diff_sets(["same_2", "dirichlet_2", "same_0125", "dirichlet_0125"])

    # lambda_search("n")

    # plot_compression_rates()
