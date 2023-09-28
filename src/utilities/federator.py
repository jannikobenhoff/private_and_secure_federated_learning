from fractions import Fraction

import tensorflow as tf
import numpy as np
import time

from keras.models import clone_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tqdm import tqdm


def active_client(prob: Fraction, max_iter: int, number_clients: int):
    active_client_matrix = np.random.choice(a=[False, True], size=(max_iter, number_clients), p=[prob, 1 - prob])
    return active_client_matrix


@tf.function
def train_step(data, label, model, loss_func):
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss_func(label, logits)
        # Add L2 regularization losses
        total_loss = loss_value + tf.reduce_sum(model.losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    return grads, loss_value


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
                     batch_size, reminder_size, client_id, number_clients, learning_rate, progress_bar,
                     initial_weights):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    client_average_grads = [tf.zeros_like(var) for var in model.trainable_variables]

    # client_model = clone_model(model)
    # client_model.set_weights(model.get_weights())

    # local training
    for _ in tf.range(local_iter):  # wie oft updates
        trainable_variables = model.trainable_variables
        iter_avg_grads = [tf.zeros_like(var) for var in trainable_variables]
        for batch, (data, label) in trainset.enumerate():
            grads, loss_value = train_step(data, label, model, loss_func)
            grads = tf.cond(tf.equal(batch, number_of_batches - 1),
                            lambda: tf.nest.map_structure(lambda x: x / batch_size * reminder_size, grads),
                            lambda: grads)

            iter_avg_grads = tf.nest.map_structure(lambda x, y: x + y, iter_avg_grads, grads)

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(label, model(data, training=False))
            # Update progress bar
            progress_bar.set_postfix({"Client": client_id, "Training loss": f"{epoch_loss_avg.result().numpy():.4f}",
                                      "Training accuracy": f"{epoch_accuracy.result().numpy():.4f}"})

        iter_avg_grads = tf.nest.map_structure(lambda x: x / number_of_batches, iter_avg_grads)
        client_average_grads = tf.nest.map_structure(lambda x, y: x + y, client_average_grads, iter_avg_grads)

        optimizer.apply_gradients(zip(iter_avg_grads, model.trainable_variables))

    # client_average_grads = [a - b for a, b in zip(client_model.trainable_variables, initial_weights)]
    # Normalize iterations
    if local_iter > 1:
        client_average_grads = tf.nest.map_structure(lambda x: x / local_iter, client_average_grads)
    # print(tf.reduce_sum([tf.reduce_sum(a) for a in client_average_grads]))

    # Gradient Compression
    compressed_data = optimizer.compress(client_average_grads, model.trainable_variables, client_id,
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

    test_loss = []
    test_acc = []
    time_per_iteration = []
    print("Local iterations:", local_iter)
    for num_iter in range(max_iter):
        active_client_number = 0

        client_data = {}

        iter_start_time = time.time()

        # lr decay
        # if num_iter in [120]:
        #     learning_rate = learning_rate * 0.1
        #     optimizer.learning_rate = learning_rate
        progress_bar_clients = tqdm(range(number_clients), desc=f"Iteration {num_iter + 1}/{max_iter}", unit="client",
                                    ncols=150)
        for k in progress_bar_clients:
            progress_bar_clients.set_description(
                f"Iteration {num_iter + 1}/{max_iter}, Client {k + 1}/{number_clients}")

            if varying_local_iter and k:
                if local_iter_type == "exponential":
                    local_iter = [np.maximum(np.round(np.random.exponential(mean[i])), 1) for i in
                                  range(number_clients)]
                elif local_iter_type == "gaussian":
                    std = 1
                    local_iter = [np.maximum(np.round(np.random.normal(mean[i], std)), 1) for i in
                                  range(number_clients)]

            if active_clients[num_iter][k]:
                if local_iter[k] < 1:
                    continue

                active_client_number += 1

                # prepare the data, shuffle, divide into batches
                data = train_data[k]
                size_of_batch = len(data)

                label = train_label[k]
                train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
                train_dataset = train_dataset.shuffle(len(label), reshuffle_each_iteration=True)
                train_dataset = train_dataset.batch(size_of_batch, drop_remainder=False)
                train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                number_of_batches = len(train_dataset)
                reminder_size = len(label) % batch_size

                # Reset Local Model with weights from last global update
                model_federator.set_weights(federator_weights)
                initial_trainable_weights = [var.numpy() for var in model_federator.trainable_variables]

                # Local Training Loop
                compressed_data = local_train_loop(
                    train_dataset, model_federator, number_of_batches, local_iter[k], loss_func, optimizer, acc_func,
                    loss_metric, batch_size, reminder_size, active_client_number, number_clients, learning_rate,
                    progress_bar_clients, initial_trainable_weights)

                client_data["client_" + str(active_client_number)] = compressed_data
                model_federator.set_weights(federator_weights)

        # Taking gradients, setting new weights, measuring validation accuracy after each epoch
        if True:  # active_client_number >= (number_clients * (0.75)):
            client_grads = optimizer.decompress(client_data, model_federator.trainable_variables)

            # Update federator weights with averaged client weights
            optimizer.apply_gradients(zip(client_grads, model_federator.trainable_variables))

            federator_weights = model_federator.get_weights()

            # val_loss_avg = tf.keras.metrics.Mean()
            # val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            # for i in range(num_class):
            #     validation_data = val_data[i]
            #     validation_label = val_label[i]
            #
            #     val_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_label)).batch(batch_size)
            #     for data, label in val_dataset:
            #         logits = model(data, training=False)
            #         val_loss_value = loss_func(label, logits)
            #         val_loss_avg.update_state(val_loss_value)
            #         val_accuracy.update_state(label, logits)

            # Evaluation on Test Data
            test_loss_avg = tf.keras.metrics.Mean()
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(batch_size)
            for data, label in test_dataset:
                logits = model(data, training=False)
                test_loss_value = loss_func(label, logits)
                test_loss_avg.update_state(test_loss_value)
                test_accuracy.update_state(label, logits)

            test_acc.append(test_accuracy.result().numpy())
            test_loss.append(test_loss_avg.result().numpy())
            time_per_iteration.append(time.time() - iter_start_time)

            print("  Test Accuracy:", f"{test_acc[-1]: .4f} | Test loss:", f"{test_loss[-1] : .4f}")
            print("  Learning rate: ", learning_rate, " | Time taken: ",
                  f"{time.time() - iter_start_time: .1f}s | {optimizer.optimizer_name} {optimizer.compression_name}")
        else:
            print('Epoch ', num_iter + 1, ' is ignored')

    if optimizer.compression is not None:
        compression_rates = [np.mean(optimizer.compression.compression_rates)]
    else:
        # If SGD was used without compression method
        compression_rates = [1]

    return model_federator, test_loss, test_acc, compression_rates, time_per_iteration
