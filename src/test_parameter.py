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
from keras.optimizers import SGD
from tensorflow import keras

from utilities import Strategy
from models.LeNet import LeNet5
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
from src.plots.plot_training_result import plot_training_result
from src.utilities.datasets import load_dataset

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=100)

    model = LeNet5(input_shape=input_shape, l2_lambda=0.0015)

    model.compile(optimizer=SGD(learning_rate=0.1), loss="SparseCategoricalCrossentropy", metrics="accuracy")

    n_samples = img_train.shape[0]

    model.fit(img_train, label_train, batch_size=n_samples, validation_data=(img_test, label_test), epochs=150)
