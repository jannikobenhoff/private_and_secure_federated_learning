import numpy as np
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

from src.utilities.datasets import load_dataset

tf.config.set_visible_devices([], 'GPU')

img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=100)


def create_model(lambda_l2):
    # create model
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape,kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(16, kernel_size=(3, 3),
                            activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.Dense(10, activation='softmax'))

    def custom_loss(y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        l2_loss = lambda_l2 * tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
        return loss + l2_loss

    model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',  #custom_loss, #
                  metrics=['accuracy'])

    return model


lambdas = [0, 0.000584928836834173]
epochs = 200
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for l2 in lambdas:
    model = create_model(l2)
    history = model.fit(img_train, label_train, validation_data=(img_test, label_test), epochs=epochs, verbose=1,
                        batch_size=32)
    train_loss.append(history.history["loss"])
    val_loss.append(history.history["val_loss"])
    train_acc.append(history.history["accuracy"])
    val_acc.append(history.history["val_accuracy"])

colors = ["blue", "r", "orange", "green", "pink", "black"]

fig, ax = plt.subplots(1, 2)
for i in range(len(lambdas)):
    print(max(val_acc[i]), np.mean(val_acc[i]), min(val_loss[i]), np.mean(val_loss[i]))

    ax[0].scatter(np.linspace(1, epochs, epochs), train_loss[i], c=colors[i])
    ax[0].plot(np.linspace(1, epochs, epochs), val_loss[i], c=colors[i])
    # ax[0].set_ylim([0, 3])

    ax[1].scatter(np.linspace(1, epochs, epochs), train_acc[i], c=colors[i])
    ax[1].plot(np.linspace(1, epochs, epochs), val_acc[i], c=colors[i])

plt.grid()
plt.show()
