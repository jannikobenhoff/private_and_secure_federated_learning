import json
from datetime import datetime
import time

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.plots import plot_convergence, plot_gaussian_process
from skopt.space import Real
from skopt.utils import use_named_args
from tensorflow import keras
from keras import models, layers, regularizers

from compressions.Atomo import Atomo
from compressions.TopK import TopK
from models.LeNet import LeNet
from optimizer.EFsignSGD import EFsignSGD
from compressions.GradientSparsification import GradientSparsification
from optimizer.MemSGD import MemSGD
from compressions.NaturalCompression import NaturalCompression
from compressions.OneBitSGD import OneBitSGD
from compressions.SparseGradient import SparseGradient
from compressions.TernGrad import TernGrad
from src.optimizer.FetchSGD import FetchSGD
from utilities.strategy import Strategy

strategies = [
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=TernGrad(clip=2.5)),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=NaturalCompression()),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=SparseGradient(drop_rate=0.99)),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=GradientSparsification(max_iter=2, k=0.004)),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=OneBitSGD()),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=TopK(k=10)),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=
    #          Atomo(sparsity_budget=2)),
    # Strategy(optimizer=EFsignSGD(learning_rate=0.001),
    #          compression=None),
    # Strategy(optimizer=FetchSGD(learning_rate=0.001, c=10000, r=1),
    #          compression=None),
    Strategy(optimizer=MemSGD(learning_rate=0.001, top_k=10),
             compression=None),
]


def train_model(strategy):
    (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()
    img_train = img_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    img_test = img_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        mode='auto',
        restore_best_weights=True,
    )

    search_space = [Real(1e-7, 0.1, "log-uniform", name='lambda_l2')]

    all_acc = []

    @use_named_args(search_space)
    def objective(**params):
        all_scores = []
        model = LeNet(num_classes=10,
                      input_shape=(28, 28, 1),
                      chosen_lambda=None, search=True).search_model(params['lambda_l2'])
        model.compile(optimizer=strategy.optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      )

        for train_index, val_index in kf.split(img_train):
            x_train, x_val = img_train[train_index], img_train[val_index]
            y_train, y_val = label_train[train_index], label_train[val_index]
            hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32,
                             verbose=0, callbacks=[early_stopping])

            all_scores.append(hist.history['val_loss'][-1])
            strategy.summary(add="val_acc: {:.4f}".format(hist.history['val_accuracy'][-1]))
        all_acc.append(hist.history['val_accuracy'][-1])
        return np.mean(all_scores)

    result = gp_minimize(objective, search_space, n_calls=3, n_initial_points=0, x0=[0.08], random_state=1)

    print("Best lambda: {}".format(result.x))
    print("Best validation loss: {}".format(result.fun))

    xiter = [x[0] for x in result.x_iters]
    dump(result, f'bayesian_result_{strategy.get_plot_title()}.pkl', store_objective=False)
    dump(all_acc, f'bayesian_acc_{strategy.get_plot_title()}.pkl')

    fig, axs = plt.subplots(3)

    plot_gaussian_process(result, ax=axs[0], show_acq_funcboolean=True)

    axs[1].scatter(xiter, all_acc)

    plot_convergence(result, ax=axs[2])

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_workers = multiprocessing.cpu_count()
    print("Using: ", num_workers, "workers")
    executor = ThreadPoolExecutor(max_workers=num_workers)

    futures = [executor.submit(train_model, strategy) for strategy in strategies]

    for future in futures:
        future.result()
