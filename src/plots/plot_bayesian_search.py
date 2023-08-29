import os

import matplotlib.pyplot as plt
import numpy as np
from skopt import load
from skopt.plots import plot_gaussian_process

if __name__ == "__main__":
    filename = "bayesian_result_SGD_cifar10_vgg11.pkl"

    result = load('../results/bayesian/' + filename)
    metrics = result["metrics"]

    xiter = [x[0] for x in result.x_iters]

    print(result.x)
    print("lambda iterations:  ", xiter)
    print("validation accuracy:    ", result.func_vals)

    # fig, axs = plt.subplots(int(np.ceil(len(xiter)/3))+1, 3, figsize=(12, 10))
    fig, axs = plt.subplots(1, 1, figsize=(9, 6))

    # axs = axs.flatten()
    plot_gaussian_process(result, ax=axs, show_acq_funcboolean=True, show_title=False)
    axs.set_title("best lambda: {:.7f}, validation accuracy: {:.3f}".format(result.x[0], -result.fun), fontsize=10)
    axs.set_xlabel("")
    axs.set_ylabel("")
    axs.set_xscale('log')

    plt.suptitle(str(metrics["args"]).replace("Namespace", "").replace("}', e", "}',\ne")[1:-1], fontsize=8)
    plt.tight_layout()
    print(os.listdir("../../"))
    # plt.savefig("../../figures/bayesian/" + filename.replace("pkl", "") + ".pdf", bbox_inches='tight')
    plt.show()
