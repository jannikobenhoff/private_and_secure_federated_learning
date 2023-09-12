import ast
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from skopt import load
from skopt.plots import plot_gaussian_process
from plot_compression_rates import get_all_files_in_directory
from plot_utils import colors, markers, names


def plot_bayesian_search(folder: str, title: str):
    directory_path = '../results/bayesian/' + folder
    all_files = get_all_files_in_directory(directory_path)

    fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    cs = iter(["g", "b", "r", "y"])

    print(len(all_files), "Files")
    plot_title = ""
    for file in all_files:

        result = load(file)

        metrics = result["metrics"]
        param = ast.literal_eval(metrics["args"].strategy)
        label_name = names[(param["optimizer"] + " " +
                            param["compression"]).replace(
            " none", "").replace(" None", "")]

        if "DS" in file:
            continue
        if title.lower() not in file.lower() and title not in (param["optimizer"] + " " +
                                                               param["compression"]).replace(
            " none", ""):
            continue

        metrics.pop("training_loss")
        metrics.pop("training_acc")
        metrics.pop("val_loss")
        metrics.pop("val_acc")
        pprint(metrics)
        print(param)

        result.func_vals = -1 * result.func_vals
        result.fun = -1 * result.fun

        param.pop("optimizer")
        param.pop("compression")
        param.pop("learning_rate")
        if param == {}:
            legend_name = label_name
        else:
            legend_name = ', '.join(
                f"{key}: {value}" for key, value in param.items() if
                value != "None")
        print(legend_name)
        plot_gaussian_process(result, ax=axs, show_acq_funcboolean=True, show_title=False,
                              **{"color": next(cs), "marker": markers[label_name], "label": legend_name})

        # axs.set_title("best lambda: {:.7f}, validation accuracy: {:.3f}".format(result.x[0], -result.fun), fontsize=10)
        plot_title = label_name

    axs.set_title(
        "Bayesian Search L2 regularization - {} - {}".format(metrics["args"].model, plot_title),
        fontsize=10, fontweight="bold")
    axs.set_xlabel("Lambda", fontsize=10)
    axs.set_ylabel("Test Accuracy", fontsize=10)
    axs.set_xscale('log')
    # axs.invert_yaxis()

    # plt.suptitle(label_name, fontsize=8)
    plt.tight_layout()
    # print(str(metrics["args"]).replace("Namespace", "").replace("}', e", "}',\ne")[1:-1])
    # plt.savefig("../../figures/bayesian_" + plot_title + ".pdf", bbox_inches='tight')
    plt.show()


def plot_bayesian_search_fed(folder: str, title: str):
    directory_path = '../results/bayesian/' + folder
    all_files = get_all_files_in_directory(directory_path)

    fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    cs = iter(["g", "b", "r", "y"])

    print(len(all_files), "Files")
    plot_title = ""
    for file in all_files:

        result = load(file)

        metrics = result["metrics"]
        param = ast.literal_eval(metrics["args"].strategy)
        label_name = names[(param["optimizer"] + " " +
                            param["compression"]).replace(
            " none", "").replace(" None", "")]

        if "DS" in file:
            continue
        if title.lower() not in file.lower() and title not in (param["optimizer"] + " " +
                                                               param["compression"]).replace(
            " none", ""):
            continue

        result.func_vals = -1 * result.func_vals
        result.fun = -1 * result.fun

        plot_gaussian_process(result, ax=axs, show_acq_funcboolean=True, show_title=False,
                              **{"color": next(cs), "marker": markers[label_name], "label": "legend_name"})

        # axs.set_title("best lambda: {:.7f}, validation accuracy: {:.3f}".format(result.x[0], -result.fun), fontsize=10)
        plot_title = label_name

    axs.set_title(
        "Bayesian Search L2 regularization - {} - {}".format(metrics["args"].model, plot_title),
        fontsize=10, fontweight="bold")
    axs.set_xlabel("Lambda", fontsize=10)
    axs.set_ylabel("Test Accuracy", fontsize=10)
    axs.set_xscale('log')
    # axs.invert_yaxis()

    # plt.suptitle(label_name, fontsize=8)
    plt.tight_layout()
    # print(str(metrics["args"]).replace("Namespace", "").replace("}', e", "}',\ne")[1:-1])
    # plt.savefig("../../figures/bayesian_" + plot_title + ".pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_bayesian_search("fed", "bayesian_result_lenet_09_12_18_34_05.pkl")

    plot_bayesian_search_fed("fed", "bayesian_result_lenet_09_12_18_34_05.pkl")
