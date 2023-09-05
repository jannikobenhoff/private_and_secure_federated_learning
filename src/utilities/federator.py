from fractions import Fraction

import keras.metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, LambdaCallback
from keras.optimizers import SGD, Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

from numpy.linalg import norm


def active_client(prob: Fraction, max_iter: int, number_clients: int):
    active_client_matrix = np.random.choice(a=[False, True], size=(max_iter, number_clients), p=[prob, 1 - prob])
    # print(active_client_matrix)

    return active_client_matrix


@tf.function
def train_step(data, label, model, loss_func):
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss_func(label, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    return grads, loss_value


@tf.function
def test_step(data, label, model, acc_func, loss_metric):
    logits = model(data, training=False)
    acc_func.update_state(label, logits)
    loss_metric.update_state(label, logits)
    acc = acc_func.result()
    loss = loss_metric.result()
    acc_func.reset_state()
    loss_metric.reset_state()
    return loss, acc


# @tf.function
def local_train_loop(trainset, model, number_of_batches, local_iter, loss_func, optimizer, acc_func, loss_metric,
                     batch_size, reminder_size, client_id, number_clients, learning_rate):
    # local_loss_federator = tf.Variable(0.0)
    # local_acc_federator = tf.Variable(0.0)
    # for data, label in trainset:
    #     loss, acc = test_step(data, label, model, acc_func, loss_metric)
    #     local_loss_federator.assign_add(loss)
    #     local_acc_federator.assign_add(acc)

    # local training
    trainable_variables = model.trainable_variables
    local_average_grads = [tf.zeros_like(var) for var in trainable_variables]
    for _ in tf.range(local_iter):

        for batch, (data, label) in trainset.enumerate():
            grads, loss_value = train_step(data, label, model, loss_func)
            grads = tf.cond(tf.equal(batch, number_of_batches - 1),
                            lambda: tf.nest.map_structure(lambda x: x / batch_size * reminder_size, grads),
                            lambda: grads)

            local_average_grads = tf.nest.map_structure(lambda x, y: x + y, local_average_grads, grads)

        local_average_grads = tf.nest.map_structure(lambda x: x / number_of_batches, local_average_grads)

    # Gradient Compression
    compressed_data = optimizer.federated_compress(local_average_grads, model.trainable_variables, client_id,
                                                   number_clients, learning_rate)
    return compressed_data


def federator(active_clients: np.array, learning_rate: float, model: Model, train_data: list, train_label: list,
              test_data: np.ndarray,
              test_label: np.ndarray, number_clients: int, max_iter: int, batch_size: int, local_iter: list,
              val_data: list,
              val_label: list, loss_func, acc_func, optimizer, loss_metric, num_sample: int, num_class: int,
              early_stopping: bool,
              es_rate: float, varying_local_iter: bool, folder_name: str, local_iter_type: str, mean=3):
    size_of_batch = batch_size

    model_federator = model  # Create returned model
    federator_weights = model_federator.get_weights()
    # client_weights = model_federator.get_weights()
    # prev_grads = np.concatenate([(x - y).reshape(-1) for x, y in zip(client_weights, federator_weights)])
    gradients = np.zeros(0)

    test_loss = np.zeros(0)
    # test_accuracy = np.zeros(0)

    # settings for early stopping
    # stop training if train_loss decreases less than es_rate over a certain number of epochs
    wait = 0
    patience = 3
    start_time = time.time()

    test_loss = []
    test_acc = []
    time_per_iteration = []
    for num_iter in range(max_iter):
        if num_iter > 0:
            print('Federated learning iteration: ', num_iter + 1, "/", max_iter, "\nTime taken:",
                  round(time.time() - start_time, 1), "s")
            start_time = time.time()
        else:
            print('Federated learning iteration: ', num_iter + 1, "Max iteration:", max_iter)
        active_client_number = 0

        client_weights = 0
        client_data = {}

        iter_start_time = time.time()
        for k in range(number_clients):

            if varying_local_iter and k:
                if local_iter_type == "exponential":
                    local_iter = [np.maximum(np.round(np.random.exponential(mean[i])), 1) for i in
                                  range(number_clients)]
                elif local_iter_type == "gaussian":
                    std = 1
                    local_iter = [np.maximum(np.round(np.random.normal(mean[i], std)), 1) for i in
                                  range(number_clients)]

            if (active_clients[num_iter][k] == True):
                active_client_number += 1

                print('Client', k + 1)

                # prepare the data, shuffle, divide into batches
                data = train_data[k]
                label = train_label[k]
                train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
                train_dataset = train_dataset.shuffle(len(label), reshuffle_each_iteration=True)  # shuffle the order
                train_dataset = train_dataset.batch(size_of_batch, drop_remainder=False)
                train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                number_of_batches = len(train_dataset)
                reminder_size = len(label) % batch_size

                # reset local model
                model_federator.set_weights(federator_weights)
                # local training loop
                # model_federator, local_loss, local_acc, local_loss_federator, local_acc_federator = local_train_loop(
                #     train_dataset, model_federator, number_of_batches, local_iter[k], loss_func, optimizer, acc_func,
                #     loss_metric, batch_size, reminder_size)
                compressed_data = local_train_loop(
                    train_dataset, model_federator, number_of_batches, local_iter[k], loss_func, optimizer, acc_func,
                    loss_metric, batch_size, reminder_size, active_client_number, number_clients, learning_rate)

                # Add client weight to all client weights
                client_weights = tf.cond(tf.equal(active_client_number, 1), lambda: model_federator.get_weights(),
                                         lambda: tf.nest.map_structure(lambda x, y: x + y, client_weights,
                                                                       model_federator.get_weights()))

                # client_grads = tf.cond(tf.equal(active_client_number, 1), lambda: compressed_data["compressed_grad"],
                #                        lambda: tf.nest.map_structure(lambda x, y: x + y, client_grads,
                #                                                      compressed_data["compressed_grad"]))

                client_data["client_" + str(active_client_number)] = compressed_data

        # Average client weights and metrics
        client_weights = tf.nest.map_structure(lambda x: x / active_client_number, client_weights)
        # client_grads = tf.nest.map_structure(lambda x: x / active_client_number, client_grads)

        # Taking gradients, setting new weights, measuring validation accuracy after each epoch
        if (active_client_number >= (number_clients * (0.75))):
            # current_grads = np.concatenate([(x - y).reshape(-1) for x, y in zip(client_weights, federator_weights)])
            # gradients = np.append(gradients, norm((current_grads - prev_grads) / (l_rate)))
            # prev_grads = current_grads
            client_grads = optimizer.federated_decompress(client_data, federator_weights, learning_rate)

            # Update federator weights with averaged client weights
            weight_index = 0
            grad_index = 0

            while weight_index < len(federator_weights) and grad_index < len(client_grads):
                if federator_weights[weight_index].shape == client_grads[grad_index].shape:
                    federator_weights[weight_index] -= client_grads[grad_index] * learning_rate
                    weight_index += 1
                    grad_index += 1
                else:
                    # If shapes don't match, try the next weight with the current gradient
                    weight_index += 1

            model_federator.set_weights(weights=federator_weights)

            # evaluation on sampled data from training data

            val_acc = tf.Variable(0, dtype=tf.float32)
            val_loss = tf.Variable(0, dtype=tf.float32)
            for i in range(num_class):
                validation_data = val_data[i]
                validation_label = val_label[i]
                val_loss_per_class, val_acc_per_class = test_step(validation_data, validation_label, model_federator,
                                                                  acc_func, loss_metric)

                val_acc.assign_add(val_acc_per_class)
                val_loss.assign_add(val_loss_per_class)
            val_acc.assign(val_acc / num_class)
            val_loss.assign(val_loss / num_class)
            print("Validation accuracy:", val_acc.numpy())
            print("Validation loss:", val_loss.numpy())

            # evaluation on test data
            t_loss, t_acc = test_step(test_data, test_label, model_federator, acc_func, loss_metric)
            print("Test accuracy:", t_acc.numpy())
            print("Test loss:", t_loss.numpy())
            print("LR:", learning_rate)
            test_acc.append(t_acc.numpy())
            test_loss.append(t_loss.numpy())
            time_per_iteration.append(time.time() - iter_start_time)
            # if early_stopping and num_iter:
            #     if test_loss[num_iter - 1] - test_loss[num_iter] < es_rate:
            #         wait += 1
            #     else:
            #         wait = 0
            #     if wait >= patience:
            #         break

        else:
            print('Epoch ', num_iter + 1, ' is ignored')

    # x_range = np.arange(1, gradients.size + 1)
    # plt.figure(2)
    # plt.plot(
    #     x_range,
    #     gradients
    # )
    # plt.title('Gradients vs Epoch')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Gradient Norm')
    # plt.grid(visible=True)
    # plt.show()
    # plt.savefig('logs/' + folder_name + '/Grad vs Epoch.png')
    # plt.figure(3, figsize=(10, 10))
    # plt.plot(x_range, train_loss, label='train')
    # plt.plot(x_range_federator, train_loss_federator[1:], label='train after updating')
    # plt.plot(x_range, validation_loss, label='validation')
    # plt.plot(x_range, test_loss, label='test')
    # plt.legend()
    # plt.title('Loss vs Epoch')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.grid(visible=True)
    # plt.savefig('Loss vs Epoch.png')

    # plt.figure(4, figsize=(10, 10))
    # plt.plot(x_range, train_accuracy, label='train')
    # plt.plot(x_range_federator, train_accuracy_federator[1:], label='train after updating')
    # plt.plot(x_range, validation_accuracy, label='validation')
    # plt.plot(x_range, test_accuracy, label='test')
    # plt.legend()
    # plt.title('Accuracy vs Epoch')
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Accuracy')
    # plt.grid(visible=True)
    # plt.savefig('Accuracy vs Epoch.png')

    compression_rates = []
    if optimizer.compression is not None:
        compression_rates = [np.mean(optimizer.compression.compression_rates)]
    elif optimizer.optimizer_name != "sgd":
        compression_rates = [np.mean(optimizer.optimizer.compression_rates)]
    else:
        compression_rates = [1]

    return model_federator, test_loss, test_acc, compression_rates, time_per_iteration
