import ast
import json
import platform
import sys
import time
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from utilities import Strategy
from models.LeNet import LeNet
from sklearn.model_selection import KFold
from keras import models, layers, regularizers

from src.compressions.Atomo import Atomo
from src.compressions.vqSGD import vqSGD
from src.models.ResNet import ResNet
from src.optimizer.FetchSGD import FetchSGD
from src.optimizer.EFsignSGD import EFsignSGD
from src.compressions.GradientSparsification import GradientSparsification
from src.compressions.bSGD import bSGD
from src.optimizer.MemSGD import MemSGD
from src.compressions.NaturalCompression import NaturalCompression
from src.compressions.OneBitSGD import OneBitSGD
from src.compressions.SparseGradient import SparseGradient
from src.compressions.TernGrad import TernGrad
from src.compressions.TopK import TopK
from src.optimizer.SGD import SGD
from src.plots.plot_training_result import plot_training_result
from src.utilities.datasets import load_dataset

if __name__ == "__main__":
    # tf.config.set_visible_devices([], 'GPU')
    # tf.config.run_functions_eagerly(run_eagerly=True)
    # tf.data.experimental.enable_debug_mode()
    #
    # img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=100)
    #
    # bucket_range = [100, 1000, 10000]
    # sparse_range = [[90, 95, 99], [950, 975, 990, 999], [9900, 9950, 9990, 9999]]
    # results = []
    # for i, buckets in enumerate(bucket_range):
    #     lower_bound = sparse_range[i]  # int(0.95 * buckets)
    #     upper_bound = buckets
    #
    #     for sparse_buckets in sparse_range[i]:  # range(lower_bound, upper_bound):
    #         print(buckets, sparse_buckets)
    #         model = LeNet(num_classes=num_classes,
    #                       input_shape=input_shape,
    #                       chosen_lambda=None).model
    #
    #         strategy = Strategy(compression=bSGD(buckets=buckets, sparse_buckets=sparse_buckets))
    #
    #         model.compile(optimizer=strategy,
    #                       loss='sparse_categorical_crossentropy',
    #                       metrics=['accuracy']
    #                       )
    #
    #         hist = model.fit(img_train, label_train, validation_data=(img_test, label_test), batch_size=32,
    #                          epochs=40, verbose=1)
    #
    #         results.append((buckets, sparse_buckets, hist.history, np.mean(strategy.compression.compression_rates)))
    #
    # print(results)
    # best_params = max(results, key=lambda x: x[2]["val_accuracy"])
    # print(
    #     f"Best parameters are buckets={best_params[0]} and sparse_buckets={best_params[1]} with performance={best_params[2]}")
    #
    # res = {
    #     "res": str(results)
    # }
    # f = open("res_param.json", "w")
    # json.dump(res, f)

    # 1000 975

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=1)

    fig, axes = plt.subplots(nrows=3, ncols=3)

    axes = axes.flatten()

    for ax in axes:
        index = np.random.randint(len(img_train))
        ax.imshow(img_train[index], cmap='gray')
        # ax.set_title(f"Label: {label_train[index]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("mnist.pdf")

    # f = open("res_param.json", "r")
    # f = json.load(f)
    # f["res"] = ast.literal_eval(f["res"])
    #
    # for b in f["res"]:
    #     print(b[0], b[1], np.mean(b[2]["val_accuracy"]))
