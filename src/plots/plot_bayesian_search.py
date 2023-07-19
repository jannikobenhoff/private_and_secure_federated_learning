import matplotlib.pyplot as plt
import os
import json
import numpy as np


if __name__ == "__main__":
    path = "../results/tuner_results/l2_regularization_sgd"

    x = []
    y = []
    y2 = []
    for file in os.listdir(path):
        if ".json" not in file:
            with open(path+"/"+file+"/trial.json", "r") as t:
                t = json.load(t)
                x.append(t["hyperparameters"]["values"]["lambda"])
                y.append(t["metrics"]["metrics"]["val_accuracy"]["observations"][0]["value"][0])
                y2.append(t["metrics"]["metrics"]["val_loss"]["observations"][0]["value"][0])


    def annot_max(x, y, axis=None):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = "lambda={:.8f}\nval_acc={:.4f}".format(xmax, ymax)
        if not axis:
            axis = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        axis.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 2)
    ax2 = fig.add_subplot(2, 1, 1)

    ax.scatter(x, y, color='blue', lw=2)
    ax2.scatter(x, y2, color='orange', lw=2)

    ax.set_xlabel("lambda")
    ax.set_ylabel("validation accuracy")
    ax2.set_ylabel("validation loss")
    ax.set_xscale('log')
    ax2.set_xscale('log')

    plt.title("Best lambda for L2-regularization")
    annot_max(x, np.array(y), axis=ax)

    plt.show()

