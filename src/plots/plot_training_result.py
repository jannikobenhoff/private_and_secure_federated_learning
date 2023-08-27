import json
import os

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from src.compressions.OneBitSGD import OneBitSGD


def plot_training_result(train_acc, train_loss, val_acc, val_loss, title, filename, save=False):
    epochs = len(train_acc)
    x = range(1, epochs + 1)

    train_acc = [float(x) for x in train_acc]
    val_acc = [float(x) for x in val_acc]
    train_loss = [float(x) for x in train_loss]
    val_loss = [float(x) for x in val_loss]

    fig = plt.figure(figsize=(12, 8))
    axes = fig.subplots(2, 2)

    # Plot Training Accuracy
    axes[0, 0].plot(x, train_acc, label='Training Accuracy', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    # Plot Training Loss
    axes[0, 1].plot(x, train_loss, label='Training Loss', marker='o', c="orange")
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Plot Validation Accuracy
    axes[1, 0].plot(x, val_acc, label='Validation Accuracy', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()

    # Plot Validation Loss
    axes[1, 1].plot(x, val_loss, label='Validation Loss', marker='o', c="orange")
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()

    plt.subplots_adjust(top=0.9)
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save:
        plt.savefig(f"../results/compression/{file}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    # file = "SGD_OneBitSGD_20230720-091040"
    for file in os.listdir("../results/compression/"):
        if "json" in file:
            file = file.replace(".json", "")
            f = open(f"../results/compression/{file}.json", "r")

            f = json.load(f)

            train_accuracy = f["train_acc"]
            training_loss = f["train_loss"]
            val_accuracy = f["val_acc"]
            vali_loss = f["val_loss"]
            title = f["strategy"]

            plot_training_result(train_accuracy, training_loss, val_accuracy, vali_loss,
                                 title, file,
                                 save=True)
