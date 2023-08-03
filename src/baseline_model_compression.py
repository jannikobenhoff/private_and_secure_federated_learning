import json
import platform
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorboard import program
from keras_tuner.tuners import BayesianOptimization
from models.LeNet import LeNet
from sklearn.model_selection import KFold
from keras import models, layers, regularizers

from src.compressions.Atomo import Atomo
from src.compressions.ScaleCom import ScaleCom
from src.compressions.vqSGD import vqSGD
from src.optimizer.FetchSGD import FetchSGD
from src.optimizer.EFsignSGD import EFsignSGD
from src.compressions.GradientSparsification import GradientSparsification
from src.optimizer.MemSGD import MemSGD
from src.compressions.NaturalCompression import NaturalCompression
from src.compressions.OneBitSGD import OneBitSGD
from src.compressions.SparseGradient import SparseGradient
from src.compressions.TernGrad import TernGrad
from src.compressions.TopK import TopK
from src.optimizer.SGD import SGD
from src.plots.plot_training_result import plot_training_result
from src.utilities.datasets import load_dataset
from src.utilities.strategy import Strategy


if __name__ == "__main__":
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist")

    # setup validation set
    img_val = img_train[-10000:]
    label_val = label_train[-10000:]
    img_train = img_train[:-10000]
    label_train = label_train[:-10000]

    chosen_lambda = 0.0001952415460342464

    model = LeNet(num_classes=10,
                  input_shape=input_shape,
                  chosen_lambda=chosen_lambda).model

    strategy = Strategy(optimizer=SGD(learning_rate=0.01),
                        compression=GradientSparsification())

    model.compile(optimizer=strategy.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    ds_train_batch = tf.data.Dataset.from_tensor_slices((img_train, label_train))
    training_data = ds_train_batch.batch(32)
    # ds_test_batch = tf.data.Dataset.from_tensor_slices((img_test, label_test))
    # testing_data = ds_test_batch.batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((img_val, label_val))
    val_dataset = val_dataset.batch(32)

    training_losses_per_epoch = []
    training_acc_per_epoch = []
    validation_losses_per_epoch = []
    validation_acc_per_epoch = []

    epochs = 10

    strategy.summary()
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch + 1))
        start_time = time.time()

        training_losses = []
        print("Steps:", len(training_data))
        for step, (x_batch_train, y_batch_train) in enumerate(training_data):
            # print("Step: ", step)
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            train_acc_metric.update_state(y_batch_train, logits)

            # Run one step of gradient descent by updating
            strategy.update_parameters(zip(grads, model.trainable_weights))
            training_losses.append(loss_value.numpy())
            # print(np.mean(strategy.compression.compression_rates), np.min(strategy.compression.compression_rates),
            #       np.max(strategy.compression.compression_rates))

        average_training_loss = np.mean(training_losses)
        training_losses_per_epoch.append(average_training_loss)

        train_acc = train_acc_metric.result()
        training_acc_per_epoch.append(train_acc.numpy())

        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss over epoch: %.4f" % (float(average_training_loss),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        validation_losses = []
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss = loss_fn(y_batch_val, val_logits)
            validation_losses.append(val_loss.numpy())

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        validation_acc_per_epoch.append(val_acc.numpy())

        val_acc_metric.reset_states()
        average_validation_loss = np.mean(validation_losses)
        validation_losses_per_epoch.append(average_validation_loss)
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (float(average_validation_loss),))

        print("Time taken: %.2fs" % (time.time() - start_time))

    file = open("results/compression/{}_{}_{}.json".format(strategy.optimizer.name, strategy.compression.name,
                                                        datetime.now().strftime("%Y%m%d-%H%M%S")), "w")
    metrics = {
        "strategy": strategy.get_plot_title(),
        "val_acc": validation_acc_per_epoch,
        "val_loss": validation_losses_per_epoch,
        "train_acc": training_acc_per_epoch,
        "train_loss": training_losses_per_epoch
    }
    json.dump(metrics, file, indent=4, default=str)
    file.close()
