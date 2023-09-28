import ast
import itertools
import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ax.set_ylim([0, 25])

    ax.set_xticklabels("")
    # ax.grid(alpha=0.4)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.35, 1),
    #           ncol=2, fancybox=False, shadow=False)
    # ax.legend(fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=5)

    plt.tight_layout()
    plt.savefig(f"../../figures/times.pdf", bbox_inches='tight')
    plt.show()


def plot_com_decom(selected_model):
    file = open("../results/times.json")
    data = json.load(file)

    split_times = {'Communication': [], 'Encoding': [],
                   'Decoding': [], "index": []}
    for method, split in data['split'].items():
        compress_time_list = split['compress'].split('/')
        decompress_time_list = split['decompress'].split('/')
        compress_time = np.mean(data[selected_model][method]) * int(compress_time_list[0]) / int(compress_time_list[1])
        decompress_time = np.mean(data[selected_model][method]) * int(decompress_time_list[0]) / int(
            decompress_time_list[1])
        print(compress_time, decompress_time)
        # split_times[method] = {
        #     'compress': compress_time,
        #     'decompress': decompress_time,
        # }
        split_times["Encoding"].append(compress_time * 70)
        split_times["Decoding"].append(decompress_time * 70)
        split_times["Communication"].append(11 * 1)
        split_times["index"].append(names[method])

    # Calculating the average times for each compression method
    # avg_times = {method: np.mean(times) for method, times in data['resnet'].items() if method in split_times}
    #
    # # Getting the compression and decompression times
    # compress_times = [avg_times[method] * split_times[method]['compress'] for method in avg_times]
    # decompress_times = [avg_times[method] * split_times[method]['decompress'] for method in avg_times]

    d = pd.DataFrame(split_times)  # , index=split_times["index"])
    print(d)
    # Bar chart
    bar_width = 0.35
    index = np.arange(len(split_times))

    ax = d.plot(kind='bar', stacked=True, edgecolor="black", color=["b", "g", "orange"], figsize=(8, 6))
    # bar1 = ax.bar(index, d, bar_width, label='Compression', stacked=True)
    # # bar2 = ax.bar(index + bar_width, decompress_times, bar_width, label='Decompression')
    #
    ax.set_ylabel('Time (s)')
    ax.set_ylim([0, 27])
    ax.set_xlabel('')
    ax.set_xticks([])
    # for minor ticks
    ax.set_xticks([], minor=True)
    ax.legend(loc="upper left")
    # ax.grid(alpha=0.4)
    rects = ax.patches

    for i, label in enumerate(split_times["index"]):
        height = split_times["Decoding"][i] + split_times["Communication"][i] + split_times["Encoding"][i] + 1
        ax.annotate(
            label,
            xy=(i, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            rotation=90
        )

    plt.tight_layout()
    # plt.show()
    plt.savefig("comp_decomp.pdf")


if __name__ == "__main__":
    # plot_avg_times(["baseline_resnet"], True)

    # plot_total_run_time()

    plot_times("lenet")

    # plot_com_decom("resnet")
