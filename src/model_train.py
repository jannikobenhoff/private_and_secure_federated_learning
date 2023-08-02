import argparse
import json
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tensorflow import keras

from models.ResNet import ResNet
from models.LeNet import LeNet
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
from optimizer.SGD import SGD

from utilities.datasets import load_dataset
from utilities.strategy import Strategy


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--strategy', type=str, help='Strategy')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--n_calls', type=int, help='Bayesian Search iterations')
    parser.add_argument('--k_fold', type=int, help='K-Fold')
    parser.add_argument('--lambda_l2', type=float, help='L2 regularization lambda')
    parser.add_argument('--bayesian_search', action='store_true', help='Apply Bayesian search')
    return parser.parse_args()


def strategy_factory(**params) -> Strategy:
    optimizer = None
    if params["optimizer"].lower() == "sgd":
        optimizer = SGD(learning_rate=params["learning_rate"])
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


def model_factory(model_name, lambda_l2, input_shape, num_classes):
    if model_name == "lenet":
        return LeNet(search=True).search_model(lambda_l2)
    elif model_name == "resnet":
        return ResNet().search_model(lambda_l2, input_shape, num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def worker(args):
    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset(args.dataset)

    # TODO early stopping, only store mean values in metrics -> if not early stopping will change metric length

    strategy_params = json.loads(args.strategy)
    strategy = strategy_factory(**strategy_params)

    kf = KFold(n_splits=args.k_fold, shuffle=True)

    if args.bayesian_search:
        print("Bayesian Search")

        search_space = [Real(1e-7, 0.1, "log-uniform", name='lambda_l2')]

        training_losses_per_epoch = []
        training_acc_per_epoch = []
        validation_losses_per_epoch = []
        validation_acc_per_epoch = []

        @use_named_args(search_space)
        def objective(**params):
            strategy = strategy_factory(**strategy_params)
            strategy.summary()

            model = model_factory(args.model.lower(), params["lambda_l2"], input_shape, num_classes)
            model.compile(optimizer=strategy.optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy']
                          )

            all_scores = []
            k_step = 0
            for train_index, val_index in kf.split(img_train):
                k_step += 1
                # Early stopping
                best_val_loss = float('inf')
                epochs_no_improve = 0
                patience = 5

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
                            logits = model(x_batch_train, training=True)
                            loss_value = loss_fn(y_batch_train, logits)

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
                    print("Epoch: {} of {}\nK-Step: {} of {}\nVal acc: {:.4f}\n"
                          "Val loss: {:.4f}\n{}\n---".format(epoch + 1, args.epochs, k_step, args.k_fold,
                                                             float(val_acc), float(average_validation_loss),
                                                             strategy.get_plot_title()))
                    # if average_validation_loss < best_val_loss:
                    #     # Save the model if you want
                    #     best_val_loss = average_validation_loss
                    #     epochs_no_improve = 0
                    # else:
                    #     epochs_no_improve += 1
                    #
                    # if epochs_no_improve == patience:
                    #     print('Early stopping!')
                    #     break
                all_scores.append(np.mean(validation_losses_per_epoch))
            return np.mean(all_scores)

        result = gp_minimize(objective, search_space, n_calls=args.n_calls, n_initial_points=0, x0=[0.08],
                             random_state=1)

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
        dump(result, f'results/bayesian/bayesian_result_{strategy.get_file_name()}.pkl', store_objective=False)
        print(f"Finished search for {strategy.get_plot_title()}")

    else:
        print("Training")

        training_losses_per_epoch = []
        training_acc_per_epoch = []
        validation_losses_per_epoch = []
        validation_acc_per_epoch = []

        strategy = strategy_factory(**strategy_params)
        strategy.summary()

        model = model_factory(args.model.lower(), args.lambda_l2, input_shape, num_classes)
        model.compile(optimizer=strategy.optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        k_step = 0
        for train_index, val_index in kf.split(img_train):
            k_step += 1
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
                        logits = model(x_batch_train, training=True)
                        loss_value = loss_fn(y_batch_train, logits)

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
                print("Epoch: {} of {}\nK-Step: {} of {}\nVal acc: {:.4f}\n"
                      "Val loss: {:.4f}\n{}\n---".format(epoch + 1, args.epochs, k_step, args.k_fold, float(val_acc),
                                                         float(average_validation_loss),
                                                         strategy.get_plot_title()))


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
