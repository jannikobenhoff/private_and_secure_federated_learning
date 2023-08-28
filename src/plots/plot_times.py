import ast
import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

names = {
    "terngrad": "TernGrad",
    "naturalcompression": "Natural Compression",
    "onebitsgd": "1-Bit SGD",
    "sparsegradient": "Sparse Gradient",
    "gradientsparsification": "Gradient Sparsification",
    "memsgd": "Sparsified SGD with Memory",
    "atomo": "Atomic Sparsification",
    "efsignsgd": "EF-SignSGD",
    "fetchsgd": "FetchSGD",
    "vqsgd": "vqSGD",
    "topk": "TopK",
    "sgd ": "SGD",
    "sgd": "SGD",
    "bsgd": "BucketSGD"
}


def plot_avg_times(parent_folder, bsgd: bool):
    def get_all_files_in_directory(root_path):
        all_files = []
        for subdir, dirs, files in os.walk(root_path):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                all_files.append(file_path)
        return all_files

    directory_path = '../results/compression/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    data = {}

    for file_path in all_files:
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

    pprint(data)
    n_models = len(data)

    bar_width = 0.05
    gap_width = 0.01
    fig, ax = plt.subplots()

    index = np.arange(n_models) / 2.5
    total_width = len(data[list(data.keys())[0]]) * (bar_width + gap_width) - gap_width

    offset = -(total_width / 2)

    sorted_methods = sorted(
        data[model].items(),
        key=lambda kv: np.mean(kv[1]), reverse=True
    )

    data = {'LeNet': dict(sorted_methods)}

    maximal = 0
    # Iterate over each model and plot
    for method_idx, (method_name, times) in enumerate(data[list(data.keys())[0]].items()):
        avg_times = [np.mean(data[model][method_name]) for model in data]
        print(avg_times)
        if avg_times[0] > maximal:
            maximal = avg_times[0]
        ax.bar(index + offset + method_idx * (bar_width + gap_width), avg_times, bar_width,
               label=names[method_name],
               edgecolor='black')

    # ax.set_xlabel('Model')
    ax.set_ylabel('Average Time per Epoch (s)')
    ax.set_ylim([0, maximal + 10])
    # ax.set_title('Average Iteration Wall-Clock Time by Model and Method')
    ax.set_xticks(offset + index + 9 * (bar_width + gap_width) / 2)
    ax.set_xticklabels(list(data.keys()))
    ax.grid(alpha=0.4)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_avg_times("baseline", True)
