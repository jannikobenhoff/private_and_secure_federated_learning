import time
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import os
from tensorboard import program
from keras_tuner.tuners import BayesianOptimization
from models.LeNet import LeNet
from sklearn.model_selection import KFold
from keras import models, layers, regularizers

from src.compressions.Atomo import Atomo
from src.optimizer.FetchSGD import FetchSGD
from src.optimizer.EFsignSGD import EFsignSGD
from src.compressions.GradientSparsification import GradientSparsification
from src.optimizer.MemSGD import MemSGD
from src.compressions.NaturalCompression import NaturalCompression
from src.compressions.OneBitSGD import OneBitSGD
from src.compressions.SparseGradient import SparseGradient
from src.compressions.TernGrad import TernGrad
from src.compressions.TopK import TopK
from src.utilities.strategy import Strategy

if __name__ == "__main__":
    logdir = "results/logs/compression/baseline_model-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

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

    model = LeNet(num_classes=10,
                  input_shape=input_shape,
                  chosen_lambda=chosen_lambda)

    strategy = Strategy(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                        compression=Atomo(sparsity_budget=2))

    model.compile(optimizer=strategy.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )

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

    strategy.summary()
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch + 1))
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(training_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            train_acc_metric.update_state(y_batch_train, logits)

            # Run one step of gradient descent by updating
            strategy.update_parameters(zip(grads, model.trainable_weights))

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
