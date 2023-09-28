import ast
import json
import tensorflow as tf
from matplotlib import pyplot as plt

from plot_utils import colors, markers, names
from plot_compression_rates import get_all_files_in_directory

bytes = {
    "lenet": 246824,
}
b_to_kb = 1024


def plot_mb_for_all(parent_folder: str, type: str):
    directory_path = f'../results/{type}/' + parent_folder
    all_files = get_all_files_in_directory(directory_path)

    compression_dict = {}
    for file_path in all_files:
        if "DS" in file_path:
            continue

        file = open(file_path, "r")
        file = json.load(file)
        strat = ast.literal_eval(file["args"]["strategy"])
        needed = round((bytes["lenet"] / file["compression_rates"][0]) / b_to_kb, 1)

        n = names[(strat["optimizer"].lower() + " " + strat["compression"].lower()).replace("none", "")]
        compression_dict[n] = needed

    fig, ax = plt.subplots()

    table_data = [[name, rate] for name, rate in compression_dict.items()]
    table = plt.table(cellText=table_data,
                      colLabels=["Compression Method", "KB per Update"],
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

    plt.show()
    # plt.savefig("compression_rates.pdf")


if __name__ == "__main__":
    plot_mb_for_all("same_2", "federated")
