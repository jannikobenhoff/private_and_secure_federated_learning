import os

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import layers, models, regularizers
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from skopt.plots import plot_gaussian_process
from skopt import dump

from src.compressions.TopK import TopK
from src.models.Alexnet import alexnet, AlexnetModel
from src.models.ResNet import ResNet
from src.optimizer.FetchSGD import FetchSGD
from src.strategy import Strategy
from src.utilities.datasets import load_dataset

tf.get_logger().setLevel('ERROR')
# tf.config.set_visible_devices([], 'GPU')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.run_functions_eagerly(run_eagerly=True)
tf.data.experimental.enable_debug_mode()

img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("cifar10", fullset=10)

lambdas = [0, 1e-4]  # 0.000584928836834173]  #0.0015, 0.01, 0.
epochs = 50
train_loss = []
train_acc = []
val_loss = []
val_acc = []

early = EarlyStopping(mode="auto", patience=6)

for l2 in lambdas:
    # from sklearn.preprocessing import OneHotEncoder
    #
    # encoder = OneHotEncoder()
    # encoder.fit(label_train)
    # label_train = encoder.transform(label_train).toarray()
    # label_test = encoder.transform(label_test).toarray()
    # model = create_model(l2)

    model = ResNet("resnet18", num_classes, lambda_l2=None)

    strat = Strategy(
        optimizer="sgd",
        compression=None,
        # optimizer="fetchsgd",
        learning_rate=0.1)

    print(strat.optimizer)
    model.compile(optimizer=strat,
                  loss='sparse_categorical_crossentropy',
                  # loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(img_train, label_train, validation_data=(img_test, label_test), epochs=epochs, verbose=2,
                        batch_size=128,
                        # batch_size=32,
                        # callbacks=[lr_callback]
                        )

    train_loss.append(history.history["loss"])
    val_loss.append(history.history["val_loss"])
    train_acc.append(history.history["accuracy"])
    val_acc.append(history.history["val_accuracy"])

colors = ["blue", "r", "orange", "green", "pink", "black"]

fig, ax = plt.subplots(1, 2)
for i in range(len(lambdas)):
    print("Max acc, mean acc, min loss, mean loss", max(val_acc[i]), np.mean(val_acc[i]), min(val_loss[i]),
          np.mean(val_loss[i]))

    ax[0].scatter(np.linspace(1, len(train_loss[i]), len(train_loss[i])), train_loss[i], c=colors[i])
    ax[0].plot(np.linspace(1, len(val_loss[i]), len(val_loss[i])), val_loss[i], c=colors[i])
    # ax[0].set_ylim([0, 3])

    ax[1].scatter(np.linspace(1, len(train_acc[i]), len(train_acc[i])), train_acc[i], c=colors[i])
    ax[1].plot(np.linspace(1, len(val_acc[i]), len(val_acc[i])), val_acc[i], c=colors[i], label=lambdas[i])

ax[1].legend()
plt.grid()
plt.show()
