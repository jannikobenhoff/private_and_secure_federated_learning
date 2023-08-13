import os

import matplotlib.pyplot as plt
import numpy as np
from skopt import load
from skopt.plots import plot_gaussian_process, plot_convergence


if __name__ == "__main__":
    filename = "bayesian_result_SGD_vqSGD_repetition100_mnist.pkl"

    result = load('../results/bayesian/'+filename)
    # metrics = load('../results/bayesian/bayesian_metrics_SGD_TopK_0108.pkl')
    # metrics = {"val_acc":[0,0,0,0,0], "args":0}
    metrics = result["metrics"]

    # print(result)
    # print(metrics)
    xiter = [x[0] for x in result.x_iters]

    # acc = np.array(metrics["val_acc"])
    # acc = acc.reshape(-1, len(xiter))
    # acc = acc.max(axis=0)
    print(result.x)
    print("lambda iterations:  ",   xiter)
    print("validation accuracy:    ", result.func_vals)

    # fig, axs = plt.subplots(int(np.ceil(len(xiter)/3))+1, 3, figsize=(12, 10))
    fig, axs = plt.subplots(1, 1, figsize=(9, 6))

    #axs = axs.flatten()
    plot_gaussian_process(result, ax=axs, show_acq_funcboolean=True, show_title=False)
    axs.set_title("best lambda: {:.7f}, validation accuracy: {:.3f}".format(result.x[0], -result.fun), fontsize=10)
    axs.set_xlabel("")
    axs.set_ylabel("")
    #axs.set_ylim([-1, -0.8])
    axs.set_xscale('log')

    # axs[-2].scatter(xiter, acc, marker="o", c="blue", s=12)
    # axs[-2].grid(color="grey")
    # axs[-2].set_title('validation accuracy, best: {:.4f}, mean: {:.4f}'.format(np.max(acc), np.mean(acc)), fontsize=10)
    # axs[-2].set_xscale('log')
    # axs[-2].set_ylim([min(acc) - 0.1, 1.02])

    # plot_convergence(result, ax=axs[-1])
    # axs[-1].set_title('Convergence plot', fontsize=10)

    for n_iter in range(0):#len(result.models)):
        plot_gaussian_process(result, ax=axs[n_iter + 1], show_title=False, n_calls=n_iter,
                              show_legend=False, show_next_point=True, show_observations=True)

        axs[n_iter + 1].scatter([xiter[n_iter]], [result.func_vals[n_iter]], c="blue", s=25, marker="*")

        axs[n_iter + 1].set_ylabel("")
        axs[n_iter + 1].set_xlabel("")
        axs[n_iter + 1].set_xscale('log')
        # axs[n_iter + 1].set_ylim(bottom=-0.1)
        axs[n_iter + 1].set_title("step {}, {:.7f}".format(n_iter, xiter[n_iter]), fontsize=10,
                                  fontweight="bold" if xiter[n_iter] == result.x[0] else"normal")

    plt.suptitle(str(metrics["args"]).replace("Namespace", "").replace("}', e", "}',\ne")[1:-1], fontsize=8)
    plt.tight_layout()
    print(os.listdir("../../"))
    # plt.savefig("../../figures/bayesian/" + filename.replace("pkl", "") + ".pdf", bbox_inches='tight')
    plt.show()

