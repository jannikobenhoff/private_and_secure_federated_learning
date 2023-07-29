import matplotlib.pyplot as plt
from skopt import load
from skopt.plots import plot_gaussian_process, plot_convergence

from src.models.LeNet import LeNet

if __name__ == "__main__":
    result = load('../results/bayesian/bayesian_result_02.pkl')
    acc = load('../results/bayesian/bayesian_acc_02.pkl')
    xiter = [x[0] for x in result.x_iters]
    print("lambda iterations:  ",   xiter)
    print("validation accuracy:", acc)
    print("validation loss:    ", result.func_vals)

    fig, axs = plt.subplots(len(xiter) + 3, figsize=(10, 10))

    plot_gaussian_process(result, ax=axs[0], show_acq_funcboolean=True, show_title=False, n_calls=-1)
    axs[0].set_title("best lambda: {:.7f}, val_loss: {:.3f}".format(result.x[0], result.fun), fontsize=10)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[0].set_xscale('log')

    axs[-2].scatter(xiter, acc)
    axs[-2].grid(color="grey")
    axs[-2].set_title('validation accuracy', fontsize=10)
    axs[-2].set_xscale('log')
    axs[-2].set_ylim([min(acc) - 0.1, 1.05])

    plot_convergence(result, ax=axs[-1])
    axs[-1].set_title('Convergence plot', fontsize=10)

    for n_iter in range(len(xiter)):
        plot_gaussian_process(result, ax=axs[n_iter + 1], show_title=False, n_calls=n_iter,
                              show_legend=False)

        axs[n_iter + 1].set_ylabel("")
        axs[n_iter + 1].set_xlabel("")
        axs[n_iter + 1].set_xscale('log')
        axs[n_iter + 1].set_ylim(bottom=-0.1)
        axs[n_iter + 1].set_title(f"step {n_iter}", fontsize=10)

    plt.tight_layout()
    plt.show()
    # plt.savefig("lambda_sgd_02.png")

    # path = "../results/tuner_results/l2_regularization_sgd"
    #
    # x = []
    # y = []
    # y2 = []
    # for file in os.listdir(path):
    #     if ".json" not in file:
    #         with open(path+"/"+file+"/trial.json", "r") as t:
    #             t = json.load(t)
    #             x.append(t["hyperparameters"]["values"]["lambda"])
    #             y.append(t["metrics"]["metrics"]["val_accuracy"]["observations"][0]["value"][0])
    #             y2.append(t["metrics"]["metrics"]["val_loss"]["observations"][0]["value"][0])
    #
    #
    # def annot_max(x, y, axis=None):
    #     xmax = x[np.argmax(y)]
    #     ymax = y.max()
    #     text = "lambda={:.8f}\nval_acc={:.4f}".format(xmax, ymax)
    #     if not axis:
    #         axis = plt.gca()
    #     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    #     arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    #     kw = dict(xycoords='data', textcoords="axes fraction",
    #               arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    #     axis.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(2, 1, 2)
    # ax2 = fig.add_subplot(2, 1, 1)
    #
    # ax.scatter(x, y, color='blue', lw=2)
    # ax2.scatter(x, y2, color='orange', lw=2)
    #
    # ax.set_xlabel("lambda")
    # ax.set_ylabel("validation accuracy")
    # ax2.set_ylabel("validation loss")
    # ax.set_xscale('log')
    # ax2.set_xscale('log')
    #
    # plt.title("Best lambda for L2-regularization")
    # annot_max(x, np.array(y), axis=ax)
    #
    # plt.show()
