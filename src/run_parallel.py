import json
from datetime import datetime
import time

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
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
from utilities.strategy import Strategy

strategies = [
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             TernGrad(clip=2.5)),
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             NaturalCompression()),
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             SparseGradient(drop_rate=0.99)),
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             GradientSparsification(max_iter=2, k=0.004)),
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             OneBitSGD()),
    Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
             compression=
             TopK(k=10)),
    # Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
    #          compression=
    #          Atomo(sparsity_budget=2)),
    Strategy(optimizer=EFsignSGD(learning_rate=0.001),
             compression=None),
    # Strategy(optimizer=MemSGD(learning_rate=0.05, rand_k=10),
    #          compression=None),
]


def train_model(strategy):
    (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()
    img_train = img_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    img_test = img_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # setup validation set
    img_val = img_train[-10000:]
    label_val = label_train[-10000:]
    img_train = img_train[:-10000]
    label_train = label_train[:-10000]

    input_shape = (28, 28, 1)
    chosen_lambda = 0.0001952415460342464

    input_shape = (28, 28, 1)
    chosen_lambda = 0.0001952415460342464

    model = LeNet(num_classes=10,
                  input_shape=input_shape,
                  chosen_lambda=chosen_lambda)

    model.compile(optimizer=strategy.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    # model.fit(img_train, label_train,
    #           epochs=20,
    #           validation_split=0.2,
    #           batch_size=32)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    ds_train_batch = tf.data.Dataset.from_tensor_slices((img_train, label_train))
    training_data = ds_train_batch.batch(32)
    ds_test_batch = tf.data.Dataset.from_tensor_slices((img_test, label_test))
    testing_data = ds_test_batch.batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((img_val, label_val))
    val_dataset = val_dataset.batch(32)

    epochs = 15

    training_losses_per_epoch = []
    training_acc_per_epoch = []
    validation_losses_per_epoch = []
    validation_acc_per_epoch = []

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch + 1))
        start_time = time.time()

        training_losses = []
        for step, (x_batch_train, y_batch_train) in enumerate(training_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            train_acc_metric.update_state(y_batch_train, logits)

            # Run one step of gradient descent by updating
            strategy.update_parameters(zip(grads, model.trainable_weights))

            training_losses.append(loss_value.numpy())

        average_training_loss = np.mean(training_losses)
        training_losses_per_epoch.append(average_training_loss)

        train_acc = train_acc_metric.result()
        training_acc_per_epoch.append(train_acc.numpy())

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

        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss over epoch: %.4f" % (float(average_training_loss),))
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (float(average_validation_loss),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    if strategy.compression is None:
        file = open("results/compression/{}_{}.json".format(strategy.optimizer.name,
                                                            datetime.now().strftime("%Y%m%d-%H%M%S")), "w")
    else:
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


num_workers = multiprocessing.cpu_count()
print("Using: ", num_workers, "workers")
executor = ThreadPoolExecutor(max_workers=num_workers)

futures = [executor.submit(train_model, strategy) for strategy in strategies]

for future in futures:
    future.result()
