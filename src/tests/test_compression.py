import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..utilities import Strategy
from ..models.LeNet import LeNet
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
    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnsit", fullset=100)

    chosen_lambda = None  # 0.0001952415460342464

    model = LeNet(num_classes=num_classes,
                  input_shape=input_shape,
                  chosen_lambda=chosen_lambda).model

    # model = ResNet("resnet18", num_classes, lambda_l2=None)

    # strategy = Strategy(compression=bSGD(buckets=100, sparse_buckets=95))
    strategy = Strategy(compression=Atomo(sparsity_budget=3, svd_rank=3))

    model.compile(optimizer=strategy,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    hist = model.fit(img_train, label_train, validation_data=(img_test, label_test), batch_size=32,
                     epochs=20, verbose=1)

    print(hist.history["loss"])
    print(hist.history["val_loss"])
