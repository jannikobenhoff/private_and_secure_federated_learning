from fractions import Fraction

import tensorflow as tf
import numpy as np
import time

from keras.models import Model


def active_client(prob: Fraction, max_iter: int, number_clients: int):
    active_client_matrix = np.random.choice(a=[False, True], size=(max_iter, number_clients), p=[prob, 1 - prob])
    return active_client_matrix


@tf.function
def train_step(data, label, model, loss_func):
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss_func(label, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    return grads, loss_value


@tf.function
def test_step2(data, label, model, acc_func, loss_metric, loss_func):
    logits = model(data, training=False)
    loss_value = loss_func(label, logits)
    # val_loss_avg.update_state(val_loss_value)
    # val_accuracy.update_state(label, logits)

    acc_func.update_state(label, logits)
    acc = acc_func.result()
    # loss = loss_metric.result()
    # loss_metric.update_state(loss)
    acc_func.reset_state()
    # loss_metric.reset_state()
    return loss_value, acc


def evaluate_on_sampled_data(val_data, val_label, model_federator, acc_func, loss_metric, loss_func, num_class):
    total_val_acc = 0
    total_val_loss = 0

    for i in range(num_class):
        current_val_loss, current_val_acc = test_step(data=val_data[i], label=val_label[i], model=model_federator,
                                                      acc_func=acc_func, loss_metric=loss_metric, loss_func=loss_func)
        total_val_acc += current_val_acc
        total_val_loss += current_val_loss

    average_val_acc = total_val_acc / num_class
    average_val_loss = total_val_loss / num_class

    return average_val_loss, average_val_acc


@tf.function
def test_step(data, label, model, acc_func, loss_metric, loss_func):
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
    compressed_data = optimizer.compress(local_average_grads, model.trainable_variables, client_id,
                                         number_clients)
    return compressed_data


def federator(active_clients: np.array, learning_rate: float, model: Model, train_data: list, train_label: list,
              test_data: np.ndarray,
              test_label: np.ndarray, number_clients: int, max_iter: int, batch_size: int, local_iter: list,
              val_data: list,
              val_label: list, loss_func, acc_func, optimizer, loss_metric, num_class: int,
              varying_local_iter: bool,
              folder_name: str, local_iter_type: str, mean=3):
    size_of_batch = batch_size

    # Initialize Federate Model and Optimizer
    model_federator = model
    federator_weights = model_federator.get_weights()

    optimizer.build(model_federator.trainable_variables, number_clients)

    # client_weights = model_federator.get_weights()
    # prev_grads = np.concatenate([(x - y).reshape(-1) for x, y in zip(client_weights, federator_weights)])

    test_loss = []
    test_acc = []
    time_per_iteration = []
    for num_iter in range(max_iter):
        if num_iter > 0:
            print('Federated learning iteration: ', num_iter + 1, "/", max_iter)
        else:
            print('Federated learning iteration: ', num_iter + 1, "Max iteration:", max_iter)
        active_client_number = 0

        client_data = {}

        iter_start_time = time.time()

        # lr decay
        # if num_iter == max_iter / 2:
        #     learning_rate = learning_rate / 5

        for k in range(number_clients):

            if varying_local_iter and k:
                if local_iter_type == "exponential":
                    local_iter = [np.maximum(np.round(np.random.exponential(mean[i])), 1) for i in
                                  range(number_clients)]
                elif local_iter_type == "gaussian":
                    std = 1
                    local_iter = [np.maximum(np.round(np.random.normal(mean[i], std)), 1) for i in
                                  range(number_clients)]

            if active_clients[num_iter][k]:
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

                # Reset Local Model
                model_federator.set_weights(federator_weights)

                # Local Training Loop
                compressed_data = local_train_loop(
                    train_dataset, model_federator, number_of_batches, local_iter[k], loss_func, optimizer, acc_func,
                    loss_metric, batch_size, reminder_size, active_client_number, number_clients, learning_rate)

                # Add client weight to all client weights
                # client_weights = tf.cond(tf.equal(active_client_number, 1), lambda: model_federator.get_weights(),
                #                          lambda: tf.nest.map_structure(lambda x, y: x + y, client_weights,
                #                                                        model_federator.get_weights()))

                client_data["client_" + str(active_client_number)] = compressed_data

        # Average client weights and metrics
        # client_weights = tf.nest.map_structure(lambda x: x / active_client_number, client_weights)

        # Taking gradients, setting new weights, measuring validation accuracy after each epoch
        if active_client_number >= (number_clients * (0.75)):
            client_grads = optimizer.decompress(client_data, federator_weights)

            # Update federator weights with averaged client weights
            weight_index = 0
            grad_index = 0

            while weight_index < len(federator_weights) and grad_index < len(client_grads):
                if federator_weights[weight_index].shape == client_grads[grad_index].shape:
                    federator_weights[weight_index] -= client_grads[grad_index] * optimizer.learning_rate
                    weight_index += 1
                    grad_index += 1
                else:
                    # If shapes don't match, try the next weight with the current gradient, happens with e.g.
                    # VGG / ResNet, because of un-trainable layers/weights
                    weight_index += 1

            model_federator.set_weights(weights=federator_weights)

            val_loss_avg = tf.keras.metrics.Mean()
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            for i in range(num_class):
                validation_data = val_data[i]
                validation_label = val_label[i]

                val_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_label)).batch(batch_size)
                for data, label in val_dataset:
                    logits = model(data, training=False)
                    val_loss_value = loss_func(label, logits)
                    val_loss_avg.update_state(val_loss_value)
                    val_accuracy.update_state(label, logits)

            # Evaluation on Test Data
            test_loss_avg = tf.keras.metrics.Mean()
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(batch_size)
            for data, label in test_dataset:
                logits = model(data, training=False)
                test_loss_value = loss_func(label, logits)
                test_loss_avg.update_state(test_loss_value)
                test_accuracy.update_state(label, logits)

            # test_loss_value, test_acc_value = test_step(test_data, test_label, model_federator, acc_func, loss_metric,
            #                                             loss_func)
            test_acc.append(test_accuracy.result().numpy())
            test_loss.append(test_loss_avg.result().numpy())
            time_per_iteration.append(time.time() - iter_start_time)

            print("Validation accuracy:", val_accuracy.result().numpy())
            print("Validation loss:", val_loss_avg.result().numpy())
            print("Test accuracy:", test_acc[-1])
            print("Test loss:", test_loss[-1])
            print("LR:", learning_rate)
            print("Time taken: ", time.time() - iter_start_time)
        else:
            print('Epoch ', num_iter + 1, ' is ignored')

    if optimizer.compression is not None:
        compression_rates = [np.mean(optimizer.compression.compression_rates)]
    elif optimizer.optimizer_name != "sgd":
        compression_rates = [np.mean(optimizer.optimizer.compression_rates)]
    else:
        # If SGD was used without compression method
        compression_rates = [1]

    return model_federator, test_loss, test_acc, compression_rates, time_per_iteration
