import ast
import os

import matplotlib.pyplot as plt
import numpy as np
from skopt import load
from skopt.plots import plot_gaussian_process
from plot_compression_rates import names

if __name__ == "__main__":
    filename = "bayesian_result_SGD_vgg11_09_03_18_54_20.pkl"

    result = load('../results/bayesian/' + filename)
    metrics = result["metrics"]
    print(len(metrics["val_loss"]))
    xiter = [x[0] for x in result.x_iters]

    print(result.x)
    print("lambda iterations:  ", xiter)
    print("validation accuracy:    ", result.func_vals)
    print()
    fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    result.func_vals = -1 * result.func_vals
    result.fun = -1 * result.fun

    # axs = axs.flatten()
    plot_gaussian_process(result, ax=axs, show_acq_funcboolean=True, show_title=False)
    # axs.set_title("best lambda: {:.7f}, validation accuracy: {:.3f}".format(result.x[0], -result.fun), fontsize=10)
    param = ast.literal_eval(metrics["args"].strategy)

    axs.set_title(
        "L2 regularization - {} - {}".format(metrics["args"].model,
                                             (names[(param["optimizer"] + " " +
                                                     param["compression"]).replace(
                                                 " none", "")])),
        fontsize=10, fontweight="bold")
    axs.set_xlabel("Lambda", fontsize=10)
    axs.set_ylabel("Test Accuracy", fontsize=10)
    axs.set_xscale('log')
    # axs.invert_yaxis()
    param.pop("optimizer")
    param.pop("compression")
    param.pop("learning_rate")
    label_name = ', '.join(
        f"{key}: {value}" for key, value in param.items() if
        value != "None")
    plt.suptitle(label_name, fontsize=8)
    plt.tight_layout()
    print(str(metrics["args"]).replace("Namespace", "").replace("}', e", "}',\ne")[1:-1])
    # plt.savefig("../../figures/bayesian/" + filename.replace("pkl", "") + ".pdf", bbox_inches='tight')
    plt.show()
