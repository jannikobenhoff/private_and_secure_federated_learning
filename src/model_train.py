import argparse
import time
from datetime import datetime
import json

from tensorflow import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from models.ResNet import ResNet
from models.LeNet import LeNet
from optimizer.SGD import SGD

from utilities.datasets import load_dataset

from compressions.TernGrad import TernGrad
from compressions.NaturalCompression import NaturalCompression
from optimizer.EFsignSGD import EFsignSGD
from optimizer.FetchSGD import FetchSGD
from optimizer.MemSGD import MemSGD
from compressions.GradientSparsification import GradientSparsification
from compressions.OneBitSGD import OneBitSGD
from compressions.SparseGradient import SparseGradient
from compressions.Atomo import Atomo
from compressions.TopK import TopK
from compressions.vqSGD import vqSGD

from utilities.strategy import Strategy


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--strategy', type=str, help='Strategy')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--n_calls', type=int, help='Bayesian Search iterations')
    parser.add_argument('--stop_patience', type=int, help='Early stopping patience')
    parser.add_argument('--k_fold', type=int, help='K-Fold')
    parser.add_argument('--lambda_l2', type=float, help='L2 regularization lambda')
    parser.add_argument('--fullset', type=float, help='% of dataset')
    parser.add_argument('--log', type=int, help='Log')
    parser.add_argument('--bayesian_search', action='store_true', help='Apply Bayesian search')
    return parser.parse_args()


def model_factory(model_name, lambda_l2, input_shape, num_classes):
    if model_name == "lenet":
        return LeNet(search=True).search_model(lambda_l2)
    elif model_name == "resnet":
        return ResNet().search_model(lambda_l2, input_shape, num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def strategy_factory(**params) -> Strategy:
    optimizer = None
    if params["optimizer"].lower() == "sgd":
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=params["learning_rate"])
    elif params["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    elif params["optimizer"].lower() == "efsignsgd":
        optimizer = EFsignSGD(learning_rate=params["learning_rate"])
    elif params["optimizer"].lower() == "fetchsgd":
        optimizer = FetchSGD(learning_rate=params["learning_rate"], c=params["c"],
                             r=params["r"])
    elif params["optimizer"].lower() == "memsgd":
        optimizer = MemSGD(learning_rate=params["learning_rate"], top_k=params["top_k"])

    if params["compression"].lower() == "terngrad":
        return Strategy(optimizer=optimizer,
                        compression=TernGrad(params["clip"]))
    elif params["compression"].lower() == "naturalcompression":
        return Strategy(optimizer=optimizer,
                        compression=NaturalCompression())
    elif params["compression"].lower() == "gradientsparsification":
        return Strategy(optimizer=optimizer,
                        compression=GradientSparsification(max_iter=params["max_iter"], k=params["k"]))
    elif params["compression"].lower() == "onebitsgd":
        return Strategy(optimizer=optimizer,
                        compression=OneBitSGD())
    elif params["compression"].lower() == "sparsegradient":
        return Strategy(optimizer=optimizer,
                        compression=SparseGradient(drop_rate=params["drop_rate"]))
    elif params["compression"].lower() == "topk":
        return Strategy(optimizer=optimizer,
                        compression=TopK(k=params["k"]))
    elif params["compression"].lower() == "vqsgd":
        return Strategy(optimizer=optimizer,
                        compression=vqSGD(repetition=params["repetitions"]))
    elif params["compression"].lower() == "atomo":
        # not working yet
        return Strategy(optimizer=optimizer,
                        compression=Atomo(sparsity_budget=params["sparsity_budget"]))
    elif params["compression"].lower() == "none":
        return Strategy(optimizer=optimizer,
                        compression=None)


def worker(args):
    tf.config.set_visible_devices([], 'GPU')

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset(args.dataset,
                                                                                          fullset=args.fullset)

    strategy_params = json.loads(args.strategy)
    strategy = strategy_factory(**strategy_params)

    kf = KFold(n_splits=args.k_fold, shuffle=True)

    if args.bayesian_search:

        print("Bayesian Search")
        search_space = [Real(1e-7, 1e-1, "log-uniform", name='lambda_l2')]

        training_losses_per_epoch = []
        training_acc_per_epoch = []
        validation_losses_per_epoch = []
        validation_acc_per_epoch = []

        n_iter_step = [0]

        @use_named_args(search_space)
        def objective(**params):
            strategy = strategy_factory(**strategy_params)
            print("Search step:", len(n_iter_step))
            n_iter_step.append(1)
            # k_step = 0
            # for train_index, val_index in kf.split(img_train):
            #     k_step += 1
            #
            #     train_images, val_images = img_train[train_index], img_train[val_index]
            #     train_labels, val_labels = label_train[train_index], label_train[val_index]
            #
            #     model = model_factory(args.model.lower(), params["lambda_l2"], input_shape, num_classes)
            #     model.compile(optimizer=strategy.optimizer,
            #                   loss='sparse_categorical_crossentropy',
            #                   metrics=['accuracy'])
            #
            #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.stop_patience,
            #                                                       verbose=1)
            #
            #     history = model.fit(train_images, train_labels, epochs=args.epochs,
            #                         validation_data=(val_images, val_labels), verbose=0,
            #                         callbacks=[early_stopping])
            #
            #     training_losses_per_epoch.append(history.history['loss'])
            #     validation_losses_per_epoch.append(history.history['val_loss'])
            #     training_acc_per_epoch.append(history.history['accuracy'])
            #     validation_acc_per_epoch.append(history.history['val_accuracy'])
            #
            #     average_training_loss = np.mean(history.history['loss'])
            #     average_validation_loss = np.mean(history.history['val_loss'])
            #     average_training_acc = np.mean(history.history['accuracy'])
            #     average_validation_acc = np.mean(history.history['val_accuracy'])
            #
            #     if args.log == 1:
            #         print("Epoch: {} of {}\nK-Step: {} of {}\nTrain acc: {:.4f}\nVal acc: {:.4f}\nTrain loss: {:.4f}\n"
            #               "Val loss: {:.4f}\n{} {}\n---".format(args.epochs, args.epochs, k_step, args.k_fold,
            #                                                     average_training_acc,
            #                                                     average_validation_acc,
            #                                                     average_training_loss, average_validation_loss,
            #                                                     strategy.get_plot_title(), params["lambda_l2"]))

            k_step = 0
            for train_index, val_index in kf.split(img_train):
                k_step += 1

                # Early stopping
                best_val_loss = float('inf')
                epochs_no_improve = 0
                patience = args.stop_patience

                strategy = strategy_factory(**strategy_params)

                model = model_factory(args.model.lower(), params["lambda_l2"], input_shape, num_classes)

                x_train, x_val = img_train[train_index], img_train[val_index]
                y_train, y_val = label_train[train_index], label_train[val_index]

                ds_train_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                training_data = ds_train_batch.batch(32)

                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                val_dataset = val_dataset.batch(32)

                loss_fn = keras.losses.SparseCategoricalCrossentropy()
                train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
                val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

                for epoch in range(args.epochs):
                    start_time = time.time()

                    training_losses = []
                    for step, (x_batch_train, y_batch_train) in enumerate(training_data):
                        with tf.GradientTape() as tape:
                            # Minibatch gradient descent
                            logits = model(x_batch_train, training=True)
                            loss_value = loss_fn(y_batch_train, logits)
                            l2_loss = params["lambda_l2"] * tf.reduce_sum(
                                [tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
                            loss_value += l2_loss

                        grads = tape.gradient(loss_value, model.trainable_weights)
                        train_acc_metric.update_state(y_batch_train, logits)

                        strategy.update_parameters(zip(grads, model.trainable_weights))

                        training_losses.append(loss_value.numpy())

                    average_training_loss = np.mean(training_losses)
                    training_losses_per_epoch.append(average_training_loss)

                    train_acc = train_acc_metric.result()
                    training_acc_per_epoch.append(train_acc.numpy())

                    train_acc_metric.reset_states()

                    validation_losses = []
                    for x_batch_val, y_batch_val in val_dataset:
                        val_logits = model(x_batch_val, training=False)
                        val_loss = loss_fn(y_batch_val, val_logits)
                        validation_losses.append(val_loss.numpy())
                        val_acc_metric.update_state(y_batch_val, val_logits)

                    val_acc = val_acc_metric.result()
                    validation_acc_per_epoch.append(val_acc.numpy())
                    val_acc_metric.reset_states()

                    average_validation_loss = np.mean(validation_losses)
                    validation_losses_per_epoch.append(average_validation_loss)

                    # compression_rates.append(np.mean(strategy.compression.compression_rates))
                    if args.log:
                        print(
                            "Epoch: {} of {}\nK-Step: {} of {}\nTrain acc: {:.4f}\nVal acc: {:.4f}\nTrain loss: {:.4f}\n"
                            "Val loss: {:.4f}\n{} {}\nTime taken: {:.2f}\n---".format(epoch + 1, args.epochs, k_step,
                                                                                       args.k_fold,
                                                                                       float(train_acc),
                                                                                       float(val_acc),
                                                                                       float(average_training_loss),
                                                                                       float(average_validation_loss),
                                                                                       strategy.get_plot_title(),
                                                                                       params["lambda_l2"],
                                                                                       time.time() - start_time))
                    if average_validation_loss < best_val_loss:
                        best_val_loss = average_validation_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print('Early stopping!')
                        break

            return - np.mean(validation_acc_per_epoch)

        # @use_named_args(search_space)
        # def objective(**params):
        #     strategy = strategy_factory(**strategy_params)
        #     strategy.summary()
        #
        #     all_scores = []
        #     k_step = 0
        #     for train_index, val_index in kf.split(img_train):
        #         k_step += 1
        #         # Early stopping
        #         best_val_loss = float('inf')
        #         epochs_no_improve = 0
        #         patience = int(args.epochs / 3)
        #
        #         model = model_factory(args.model.lower(), params["lambda_l2"], input_shape, num_classes)
        #         # model.compile(optimizer=strategy.optimizer,
        #         #               loss='sparse_categorical_crossentropy',
        #         #               metrics=['accuracy']
        #         #               )
        #
        #         x_train, x_val = img_train[train_index], img_train[val_index]
        #         y_train, y_val = label_train[train_index], label_train[val_index]
        #
        #         ds_train_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #         training_data = ds_train_batch.batch(32)
        #
        #         val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        #         val_dataset = val_dataset.batch(32)
        #
        #         loss_fn = keras.losses.SparseCategoricalCrossentropy()
        #         train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        #         val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        #
        #         for epoch in range(args.epochs):
        #             training_losses = []
        #             for step, (x_batch_train, y_batch_train) in enumerate(training_data):
        #
        #                 with tf.GradientTape() as tape:
        #                     # Minibatch gradient descent
        #                     logits = model(x_batch_train, training=True)
        #                     loss_value = loss_fn(y_batch_train, logits)
        #                     l2_loss = params["lambda_l2"] * tf.reduce_sum(
        #                         [tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
        #                     loss_value += l2_loss
        #
        #                 grads = tape.gradient(loss_value, model.trainable_weights)
        #                 train_acc_metric.update_state(y_batch_train, logits)
        #
        #                 strategy.update_parameters(zip(grads, model.trainable_weights))
        #
        #                 training_losses.append(loss_value.numpy())
        #
        #             average_training_loss = np.mean(training_losses)
        #             training_losses_per_epoch.append(average_training_loss)
        #
        #             train_acc = train_acc_metric.result()
        #             training_acc_per_epoch.append(train_acc.numpy())
        #
        #             train_acc_metric.reset_states()
        #
        #             validation_losses = []
        #             for x_batch_val, y_batch_val in val_dataset:
        #                 val_logits = model(x_batch_val, training=False)
        #                 val_loss = loss_fn(y_batch_val, val_logits)
        #                 validation_losses.append(val_loss.numpy())
        #                 val_acc_metric.update_state(y_batch_val, val_logits)
        #
        #             val_acc = val_acc_metric.result()
        #             validation_acc_per_epoch.append(val_acc.numpy())
        #             val_acc_metric.reset_states()
        #
        #             average_validation_loss = np.mean(validation_losses)
        #             validation_losses_per_epoch.append(average_validation_loss)
        #
        #             if args.log:
        #                 print("Epoch: {} of {}\nK-Step: {} of {}\nTrain acc: {:.4f}\nVal acc: {:.4f}\nTrain loss: {:.4f}\n"
        #                       "Val loss: {:.4f}\n{} {}\n---".format(epoch + 1, args.epochs, k_step, args.k_fold,float(train_acc),
        #                                                          float(val_acc), float(average_training_loss), float(average_validation_loss),
        #                                                          strategy.get_plot_title(), params["lambda_l2"]))
        #             if average_validation_loss < best_val_loss:
        #                 best_val_loss = average_validation_loss
        #                 epochs_no_improve = 0
        #             else:
        #                 epochs_no_improve += 1
        #             if epochs_no_improve == patience:
        #                 print('Early stopping!')
        #                 break
        #         all_scores.append(np.mean(validation_losses_per_epoch))
        #     return np.mean(all_scores)

        # result = gp_minimize(objective, search_space, n_calls=args.n_calls,  x0=[[1e-7], [1e-4], [0.1]], #n_initial_points=0,
        #                      random_state=1)
        result = gp_minimize(objective, search_space, n_calls=args.n_calls, acq_func='EI',  # kappa=1,
                             # x0=[[1e-6], [1e-4], [1e-2]],
                             # n_random_starts=3,
                             # n_jobs=3,
                             random_state=45
                             )

        print("Best lambda: {}".format(result.x))
        print("Best validation loss: {}".format(result.fun))
        metrics = {
            "training_loss": training_losses_per_epoch,
            "training_acc": training_acc_per_epoch,
            "val_loss": validation_losses_per_epoch,
            "val_acc": validation_acc_per_epoch,
            "args": args
        }
        result["metrics"] = metrics
        dump(result, f'results/bayesian/bayesian_result_{strategy.get_file_name()}_{args.dataset}.pkl',
             store_objective=False)
        print(f"Finished search for {strategy.get_plot_title()}")

    else:
        print("Training")

        training_losses_per_epoch = []
        training_acc_per_epoch = []
        validation_losses_per_epoch = []
        validation_acc_per_epoch = []
        compression_rates = []

        strategy.summary()

        k_step = -1
        for train_index, val_index in kf.split(img_train):
            training_losses_per_epoch.append([])
            training_acc_per_epoch.append([])
            validation_losses_per_epoch.append([])
            validation_acc_per_epoch.append([])
            compression_rates.append([])

            k_step += 1

            # Early stopping
            best_val_loss = float('inf')
            epochs_no_improve = 0
            patience = args.stop_patience

            strategy = strategy_factory(**strategy_params)

            model = model_factory(args.model.lower(), args.lambda_l2, input_shape, num_classes)

            x_train, x_val = img_train[train_index], img_train[val_index]
            y_train, y_val = label_train[train_index], label_train[val_index]

            ds_train_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            training_data = ds_train_batch.batch(32)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(32)

            loss_fn = keras.losses.SparseCategoricalCrossentropy()
            train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
            val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

            for epoch in range(args.epochs):
                training_losses = []
                for step, (x_batch_train, y_batch_train) in enumerate(training_data):
                    with tf.GradientTape() as tape:
                        # Minibatch gradient descent
                        logits = model(x_batch_train, training=True)
                        loss_value = loss_fn(y_batch_train, logits)
                        l2_loss = args.lambda_l2 * tf.reduce_sum(
                            [tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
                        loss_value += l2_loss

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    train_acc_metric.update_state(y_batch_train, logits)

                    strategy.update_parameters(zip(grads, model.trainable_weights))

                    training_losses.append(loss_value.numpy())

                average_training_loss = np.mean(training_losses)
                training_losses_per_epoch[k_step].append(average_training_loss)

                train_acc = train_acc_metric.result()
                training_acc_per_epoch[k_step].append(train_acc.numpy())

                train_acc_metric.reset_states()

                validation_losses = []
                for x_batch_val, y_batch_val in val_dataset:
                    val_logits = model(x_batch_val, training=False)
                    val_loss = loss_fn(y_batch_val, val_logits)
                    validation_losses.append(val_loss.numpy())
                    val_acc_metric.update_state(y_batch_val, val_logits)

                val_acc = val_acc_metric.result()
                validation_acc_per_epoch[k_step].append(val_acc.numpy())
                val_acc_metric.reset_states()

                average_validation_loss = np.mean(validation_losses)
                validation_losses_per_epoch[k_step].append(average_validation_loss)

                if strategy.compression is not None:
                    compression_rates[k_step].append(np.mean(strategy.compression.compression_rates))

                if args.log:
                    print("Epoch: {} of {}\nK-Step: {} of {}\nTrain acc: {:.4f}\nVal acc: {:.4f}\nTrain loss: {:.4f}\n"
                          "Val loss: {:.4f}\n{} {}\n---".format(epoch + 1, args.epochs, k_step + 1, args.k_fold,
                                                                float(train_acc),
                                                                float(val_acc), float(average_training_loss),
                                                                float(average_validation_loss),
                                                                strategy.get_plot_title(), args.lambda_l2))
                if average_validation_loss < best_val_loss:
                    best_val_loss = average_validation_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break
            # model = model_factory(args.model.lower(), args.lambda_l2, input_shape, num_classes)
            # model.compile(optimizer=strategy.optimizer,
            #               loss='sparse_categorical_crossentropy',
            #               metrics=['accuracy'])
            #
            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.stop_patience,
            #                                                   verbose=1)
            #
            # history = model.fit(train_images, train_labels, epochs=args.epochs,
            #                     validation_data=(val_images, val_labels), verbose=0,
            #                     callbacks=[early_stopping])
            #
            # training_losses_per_epoch.append(history.history['loss'])
            # validation_losses_per_epoch.append(history.history['val_loss'])
            # training_acc_per_epoch.append(history.history['accuracy'])
            # validation_acc_per_epoch.append(history.history['val_accuracy'])
            #
            # average_training_loss = np.mean(history.history['loss'])
            # average_validation_loss = np.mean(history.history['val_loss'])
            # average_training_acc = np.mean(history.history['accuracy'])
            # average_validation_acc = np.mean(history.history['val_accuracy'])
            #
            # compression_rates.append(strategy.compression.compression_rates)

            # if args.log == 1:
            #     print("Epoch: {} of {}\nK-Step: {} of {}\nTrain acc: {:.4f}\nVal acc: {:.4f}\nTrain loss: {:.4f}\n"
            #           "Val loss: {:.4f}\n{} {}\n---".format(args.epochs, args.epochs, k_step, args.k_fold,
            #                                                 average_training_acc,
            #                                                 validation_acc_per_epoch,
            #                                                 average_training_loss, average_validation_loss,
            #                                                 strategy.get_plot_title(), args.lambda_l2))

        metrics = {
            "training_loss": str(training_losses_per_epoch),
            "training_acc": str(training_acc_per_epoch),
            "val_loss": str(validation_losses_per_epoch),
            "val_acc": str(validation_acc_per_epoch),
            "args": vars(args),
            "compression_rates": compression_rates
        }

        file = open('results/compression/training_{}_{}_'
                    '{}.json'.format(strategy.get_file_name(), args.dataset, datetime.now().strftime('%m_%d_%H_%M')),
                    "w")
        json.dump(metrics, file, indent=4)
        file.close()
        print(f"Finished search for {strategy.get_plot_title()}")


def main():
    args = parse_args()
    worker(args)
    # tasks = []
    # for model in args.models:
    #     task_args = argparse.Namespace()
    #     task_args.model = model
    #     task_args.dataset = args.dataset
    #     task_args.optimizer = args.optimizer
    #     task_args.compression = args.compression
    #     task_args.bayesian_search = args.bayesian_search
    #     tasks.append(task_args)

    # num_workers = multiprocessing.cpu_count()
    # print("Using: ", num_workers, "workers")
    # executor = ThreadPoolExecutor(max_workers=num_workers)
    #
    # futures = [executor.submit(train_model, strategy) for strategy in strategies]
    #
    # for future in futures:
    #     future.result()

    # with Pool(processes=len(tasks)) as pool:
    #     results = pool.map(worker, tasks)
    #
    # for result in results:
    #     print(result)


if __name__ == "__main__":
    main()
