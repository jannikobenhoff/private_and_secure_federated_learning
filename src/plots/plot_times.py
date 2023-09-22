import ast
import itertools
import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import colors, names


def get_all_files_in_directory(root_path):
    all_files = []
    for subdir, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            all_files.append(file_path)
    return all_files


def plot_avg_times(parent_folders, bsgd: bool):
    all_files = []
    for parent_folder in parent_folders:
        directory_path = '../results/compression/' + parent_folder
        all_files += get_all_files_in_directory(directory_path)

    data = {}
    model = ""
    for file_path in all_files:
        if "DS_" in file_path:
            continue
        if "Bucket" in file_path and not bsgd:
            continue

        file = open(file_path, "r")
        file = json.load(file)
        model = file["args"]["model"]

        strat = ast.literal_eval(file["args"]["strategy"])
        name = f"{strat['optimizer']} {strat['compression']}"
        name = (name.replace("none", "") if name[-4:] == "none" else name[4:]).replace(" ", "").lower()
        if model not in data:
            data[model] = {}
        if "time_per_epoch" not in file:
            continue

        if name not in data[model]:
            data[model][name] = file["time_per_epoch"]
        elif np.mean(file["time_per_epoch"]) < np.mean(data[model][name]):
            data[model][name] = file["time_per_epoch"]

    n_models = len(data)
    print(n_models)

    bar_width = 0.05
    gap_width = 0.01
    fig, ax = plt.subplots()

    index = np.arange(n_models) / 2.5
    total_width = 2 * len(data[list(data.keys())[0]]) * (bar_width + gap_width) - gap_width

    offset = -(total_width / 2)

    for model in data:
        sorted_methods = sorted(
            data[model].items(),
            key=lambda kv: np.mean(kv[1]), reverse=True
        )

        data[model] = dict(sorted_methods)
    print(data.keys())

    maximal = 0

    # Iterate over each model and plot
    for m, model in enumerate(data):
        print("as")
        for method_idx, (method_name, times) in enumerate(data[model].items()):
            print(method_name)
            avg_times = [np.mean(data[model][method_name])]
            if avg_times[0] > maximal:
                maximal = avg_times[0]
            ax.bar(m * 5 + index + offset + method_idx * (bar_width + gap_width), avg_times, bar_width,
                   label=names[method_name],
                   edgecolor='black', color=colors[names[method_name]])

    ax.set_ylabel('Average Time per Epoch (s)')
    ax.set_ylim([0, maximal + 10])

    # ax.set_xticks(offset + index + len(data[model]) * (bar_width + gap_width) / 2)
    ax.set_xticks(offset + index + (len(data[model]) - 1) * (bar_width + gap_width) / 2)
    ax.set_xticklabels(list(data.keys()))
    ax.grid(alpha=0.4)

    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"../../figures/times_{parent_folder}.pdf", bbox_inches='tight')
    plt.show()


def plot_total_run_time():
    directory_path = '../results/'
    all_files = get_all_files_in_directory(directory_path)
    total_time = 0

    for file_path in all_files:
        if "DS" in file_path or ".pkl" in file_path:
            continue
        file = open(file_path, "r")
        file = json.load(file)
        if "time_per_epoch" in file:
            total_time += np.sum(file["time_per_epoch"])
        elif "times" in file:
            total_time += np.sum(file["times"]["federator"])
    print("Total Run Time:\n", round(total_time / 3600, 1), "h\n", round(total_time / 3600 / 24, 1), "days")


def plot_times(selected_model):
    file = open("../results/times.json")
    file = json.load(file)

    bar_width = 0.05
    gap_width = 0.01

    index = np.arange(1) / 2.5
    total_width = 2 * 12 * (bar_width + gap_width) - gap_width

    offset = -(total_width / 2)

    data = {}
    for method in file[selected_model]:
        data[method] = np.mean(file[selected_model][method])

    data = dict(sorted(data.items(), key=lambda x: x[1]))
    print(data)
    # sorted_methods = sorted(
    #     data.items(),
    #     key=lambda kv: np.mean(kv), reverse=True
    # )
    # data = sorted_methods
    # print(data.keys())

    maximal = 0
    # Iterate over each model and plot

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for method_idx, (method_name, times) in enumerate(data.items()):
        print(method_name)
        avg_times = [np.mean(data[method_name])]
        if avg_times[0] > maximal:
            maximal = avg_times[0]
        ax.bar(index + offset + method_idx * (bar_width + gap_width), avg_times, bar_width,
               label=names[method_name],
               edgecolor='black', color=colors[names[method_name]])

    ax.set_ylabel('Average Time per Global Iteration (s)')
    ax.set_ylim([0, maximal + 5])

    # ax.set_xticks(offset + index + len(data[model]) * (bar_width + gap_width) / 2)
    # ax.set_xticks(offset + index + (len(data) - 1) * (bar_width + gap_width) / 2)
    # ax.set_xticklabels(list(data.keys()))
    ax.set_xticklabels("")
    ax.grid(alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=False)
    # ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"../../figures/times.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_avg_times(["baseline_resnet"], True)

    # plot_total_run_time()

    plot_times("resnet")
