import os
import time
from datetime import datetime
import json

from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from models.LeNet import LeNet, LeNet5
from models.ResNet import ResNet, resnet
from models.DenseNet import DenseNet
from compressions.bSGD import bSGD
from models.VGG import VGG
from models.MobileNetV2 import MobileNetV2
from utilities.parameters import parse_args
from utilities import Strategy

from utilities.custom_callbacks import step_decay
from utilities.datasets import load_dataset

from compressions.TernGrad import TernGrad
from compressions.NaturalCompression import NaturalCompression

from compressions.GradientSparsification import GradientSparsification
from compressions.OneBitSGD import OneBitSGD
from compressions.SparseGradient import SparseGradient
from compressions.Atomo import Atomo
from compressions.TopK import TopK
from compressions.vqSGD import vqSGD


def model_factory(model_name, lambda_l2, input_shape, num_classes):
    print(f'==> Building {model_name.upper()} model..')
    if model_name == "lenet":
        model = LeNet(input_shape=input_shape, num_classes=num_classes, l2_lambda=lambda_l2)
        model = LeNet5(input_shape=input_shape, l2_lambda=lambda_l2)
        model.build(input_shape=input_shape)
        return model
    elif model_name == "resnet18":
        model = ResNet("resnet18", num_classes, lambda_l2=lambda_l2)
        model = resnet(num_filters=64, size=18, input_shape=(32, 32, 3), lambda_l2=lambda_l2)
        return model
    elif model_name == "mobilenet":
        model = MobileNetV2(num_classes, lambda_l2=lambda_l2)
        return model
    elif model_name == "vgg11":
        model = VGG(vgg_name="VGG11", l2_lambda=lambda_l2, num_classes=num_classes)
        model.build(input_shape=(None, 32, 32, 3))
        return model
    elif model_name == "densenet":
        model = DenseNet('densenet121', num_classes)  # , lambda_l2=lambda_l2)
        return model
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def strategy_factory(**params) -> Strategy:
    if params["compression"].lower() == "terngrad":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=TernGrad(params["clip"]))
    elif params["compression"].lower() == "bsgd":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=bSGD(buckets=params["buckets"], sparse_buckets=params["sparse_buckets"]))
    elif params["compression"].lower() == "naturalcompression":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=NaturalCompression())
    elif params["compression"].lower() == "gradientsparsification":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=GradientSparsification(max_iter=params["max_iter"], k=params["k"]))
    elif params["compression"].lower() == "onebitsgd":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=OneBitSGD())
    elif params["compression"].lower() == "sparsegradient":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=SparseGradient(drop_rate=params["drop_rate"]))
    elif params["compression"].lower() == "topk":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=TopK(k=params["k"]))
    elif params["compression"].lower() == "vqsgd":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=vqSGD(repetition=params["repetition"]))
    elif params["compression"].lower() == "atomo":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=Atomo(svd_rank=params["svd_rank"]))
    elif params["optimizer"].lower() == "sgdm":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=None, optimizer="sgd", momentum=params["momentum"])
    elif params["compression"].lower() == "none":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=None, optimizer=params["optimizer"].lower())


def get_l2_lambda(args, fed=False, **params) -> float:
    lambdas = None
    if fed:
        if args.model.lower() == "lenet":
            lambdas = json.load(open("../results/lambda_lookup_federated.json", "r"))
        # elif args.model.lower() == "resnet18":
        #     lambdas = json.load(open("../results/lambda_lookup_resnet18.json", "r"))
        # elif args.model.lower() == "vgg11":
        #     lambdas = json.load(open("../results/lambda_lookup_vgg11.json", "r"))
        setup = args.local_iter_type + str(args.beta)
        return lambdas["sgd"][setup]

    else:
        if args.model.lower() == "lenet":
            lambdas = json.load(open("../results/lambda_lookup.json", "r"))
        elif args.model.lower() == "resnet18":
            lambdas = json.load(open("../results/lambda_lookup_resnet18.json", "r"))
        elif args.model.lower() == "vgg11":
            lambdas = json.load(open("../results/lambda_lookup_vgg11.json", "r"))

        opt = params["optimizer"]
        comp = params["compression"]

        if (opt in ["efsignsgd", "sgd"] and comp in ["none", "federated"]) or comp in ["onebitsgd",
                                                                                       "naturalcompression"]:
            return lambdas[opt][comp]

        keys = [k for k in params.keys() if k != "compression" and k != "optimizer" and k != "learning_rate"]
        first_key = list(lambdas[opt][comp].keys())[0]
        first_key_value = lambdas[opt][comp][first_key][str(params[first_key])]

        if type(first_key_value) != float:
            second_key = list(first_key_value.keys())[0]
            if second_key in keys:
                second_key_value = first_key_value[second_key][str(params[second_key])]
                return second_key_value
        return first_key_value


@tf.function
def train_step(data, label, model, loss_func):
    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss_value = loss_func(label, logits)
        # Add L2 regularization losses
        total_loss = loss_value + tf.reduce_sum(model.losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
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


def lr_scheduler(optimizer, epoch, drop_factor, drop_epochs, min_lr):
    new_lr = step_decay(epoch, optimizer.learning_rate.numpy(), drop_factor, drop_epochs, min_lr)
    optimizer.learning_rate.assign(new_lr)


def train_model(train_images, train_labels, val_images, val_labels, lambda_l2, input_shape, num_classes,
                strategy_params, args):
    model = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)
    strategy = strategy_factory(**strategy_params)
    strategy.summary()

    BATCH_SIZE = 64
    initial_lr = strategy_params["learning_rate"]
    drop_factor = 0.1
    drop_epochs = []
    min_lr = initial_lr * 0.1 * 0.1

    if args.dataset == "cifar10" and args.model.lower() == "resnet18":
        BATCH_SIZE = 256  # 128
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.1
        drop_epochs = [20]  # [15]
        min_lr = initial_lr * 0.1 * 0.1

    elif args.dataset == "cifar10" and args.model.lower() == "vgg11":
        BATCH_SIZE = 128
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.2
        drop_epochs = [15, 30]
        min_lr = initial_lr * 0.1 * 0.1

    print("BATCH SIZE:", BATCH_SIZE)
    optimizer = strategy
    optimizer.build(model.trainable_variables)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE).shuffle(
        len(train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE)

    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []
    time_history = []

    best_val_acc = float(0)
    patience_counter = 0

    try:
        for epoch in range(args.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop over batches
            progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=150)
            epoch_start_time = time.time()
            for data, label in progress_bar:
                grads, loss_value = train_step(data, label, model, loss_func)

                compressed_data = optimizer.compress(grads, model.trainable_variables)

                if compressed_data["needs_decompress"]:
                    decompressed_grads = optimizer.decompress(compressed_data, model.trainable_variables)
                else:
                    decompressed_grads = compressed_data["compressed_grads"]
                optimizer.apply_gradients(zip(decompressed_grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(label, model(data, training=False))
                # Update progress bar
                progress_bar.set_postfix({"Training loss": f"{epoch_loss_avg.result().numpy():.4f}",
                                          "Training accuracy": f"{epoch_accuracy.result().numpy():.4f}",
                                          "Compression ratio": f"{np.mean(optimizer.compression_rates()):.2f}"})

            # End of epoch
            time_history.append(time.time() - epoch_start_time)

            train_loss_results.append(epoch_loss_avg.result().numpy())
            train_accuracy_results.append(epoch_accuracy.result().numpy())

            print("   Train Loss:", f"{epoch_loss_avg.result().numpy(): .4f}",
                  " | Train Accuracy:", f"{epoch_accuracy.result().numpy(): .4f}",
                  "| Time per Epoch:", f"{time_history[-1]:.1f}s")

            # Validation loop
            val_loss_avg = tf.keras.metrics.Mean()
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            for data, label in test_dataset:
                logits = model(data, training=False)
                val_loss_value = loss_func(label, logits)
                val_loss_avg.update_state(val_loss_value)
                val_accuracy.update_state(label, logits)

            val_loss_results.append(val_loss_avg.result().numpy())
            val_accuracy_results.append(val_accuracy.result().numpy())

            print("   Test Loss: ", f"{val_loss_avg.result().numpy(): .4f}",
                  " | Test Accuracy: ", f"{val_accuracy.result().numpy(): .4f}", "| Learning Rate:",
                  optimizer.learning_rate.numpy(), f"| {optimizer.optimizer_name} {optimizer.compression_name}")

            # Early Stopping Check
            if val_accuracy.result() > best_val_acc:
                best_val_acc = val_accuracy.result()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > args.stop_patience:
                    print("Early stopping...")
                    break

            # Adjust learning rate
            lr_scheduler(optimizer=optimizer, epoch=epoch, drop_epochs=drop_epochs,
                         drop_factor=drop_factor, min_lr=min_lr)
            interrupt = False
    except KeyboardInterrupt:
        print('Interrupting..')
        interrupt = True
    finally:
        history = {
            'loss': train_loss_results,
            'accuracy': train_accuracy_results,
            'val_loss': val_loss_results,
            'val_accuracy': val_accuracy_results
        }
        train_metrics = {"batch_size": BATCH_SIZE, "lr_decay": drop_epochs, "drop_factor": drop_factor,
                         "min_lr": min_lr}
    return history, strategy, time_history, train_metrics, interrupt


def train_model_non_mini_batch(train_images, train_labels, val_images, val_labels, lambda_l2, input_shape, num_classes,
                               strategy_params, args):
    model = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)
    strategy = strategy_factory(**strategy_params)
    strategy.summary()

    BATCH_SIZE = 256
    initial_lr = strategy_params["learning_rate"]
    drop_factor = 0.1
    drop_epochs = [0]
    min_lr = initial_lr * 0.1 * 0.1

    if args.dataset == "cifar10" and args.model.lower() == "resnet18":
        BATCH_SIZE = 128  # 256
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.1
        drop_epochs = [10, 20]
        min_lr = initial_lr * 0.1 * 0.1 * 0.1

    elif args.dataset == "cifar10" and args.model.lower() == "vgg11":
        BATCH_SIZE = 128  # 128
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.2  # 0.2
        drop_epochs = [6, 15]  # [15, 30]
        min_lr = initial_lr * 0.1 * 0.1

    print("BATCH SIZE:", BATCH_SIZE)
    optimizer = strategy
    optimizer.build(model.trainable_variables)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE).shuffle(
        len(train_images))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE)

    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []
    time_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop over batches
        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=150)
        epoch_start_time = time.time()
        iter_avg_grads = [tf.zeros_like(var) for var in model.trainable_variables]
        for data, label in progress_bar:
            grads, loss_value = train_step(data, label, model, loss_func)

            iter_avg_grads = tf.nest.map_structure(lambda x, y: x + y, iter_avg_grads, grads)

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(label, model(data, training=False))
            # Update progress bar
            progress_bar.set_postfix({"Training loss": f"{epoch_loss_avg.result().numpy():.4f}",
                                      "Training accuracy": f"{epoch_accuracy.result().numpy():.4f}",
                                      "Compression ratio": f"{np.mean(optimizer.compression_rates()):.2f}"})

        iter_avg_grads = tf.nest.map_structure(lambda x: x / len(train_dataset), iter_avg_grads)

        compressed_data = optimizer.compress(iter_avg_grads, model.trainable_variables)

        if compressed_data["needs_decompress"]:
            decompressed_grads = optimizer.decompress(compressed_data, model.trainable_variables)
        else:
            decompressed_grads = compressed_data["compressed_grads"]
        optimizer.apply_gradients(zip(decompressed_grads, model.trainable_variables))

        # End of epoch
        time_history.append(time.time() - epoch_start_time)

        train_loss_results.append(epoch_loss_avg.result().numpy())
        train_accuracy_results.append(epoch_accuracy.result().numpy())

        print("   Train Loss:     ", f"{epoch_loss_avg.result().numpy(): .4f}",
              " | Train Accuracy:     ", f"{epoch_accuracy.result().numpy(): .4f}",
              "| Time per Epoch:", f"{time_history[-1]:.1f}s")

        # Validation loop
        val_loss_avg = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for data, label in val_dataset:
            logits = model(data, training=False)
            val_loss_value = loss_func(label, logits)
            val_loss_avg.update_state(val_loss_value)
            val_accuracy.update_state(label, logits)

        val_loss_results.append(val_loss_avg.result().numpy())
        val_accuracy_results.append(val_accuracy.result().numpy())

        print("   Validation Loss:", f"{val_loss_avg.result().numpy(): .4f}",
              " | Validation Accuracy:", f"{val_accuracy.result().numpy(): .4f}", "| Learning Rate:",
              optimizer.learning_rate.numpy())

        # Early Stopping Check
        if val_loss_avg.result() < best_val_loss:
            best_val_loss = val_loss_avg.result()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > args.stop_patience:
                print("Early stopping...")
                break

        # Adjust learning rate
        lr_scheduler(optimizer=optimizer, epoch=epoch, drop_epochs=drop_epochs,
                     drop_factor=drop_factor, min_lr=min_lr)

    history = {
        'loss': train_loss_results,
        'accuracy': train_accuracy_results,
        'val_loss': val_loss_results,
        'val_accuracy': val_accuracy_results
    }
    train_metrics = {"batch_size": BATCH_SIZE, "lr_decay": drop_epochs, "drop_factor": drop_factor, "min_lr": min_lr}
    return history, strategy, time_history, train_metrics


def worker(args):
    print("WORKER")
    seed_value = 100
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)

    if args.gpu != 1:
        print("==> Running on CPU")
        tf.config.set_visible_devices([], 'GPU')
    else:
        print("==> Running on GPU")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset(args.dataset,
                                                                                          fullset=args.fullset)
    strategy_params = json.loads(args.strategy)
    strategy = strategy_factory(**strategy_params)

    if args.bayesian_search:
        print("--- Bayesian Search ---")
        search_space = [Real(1e-5, 1e0, "log-uniform", name='lambda_l2')]
        train_loss = []
        val_loss = []
        val_acc = []
        train_acc = []
        train_metrics_list = [{}]
        n_iter_step = [0]

        strategy = strategy_factory(**strategy_params)
        strategy.summary()

        @use_named_args(search_space)
        def objective(**params):
            print("Search step:", len(n_iter_step), "Lambda:", params["lambda_l2"])
            search_step_start_time = time.time()

            training_losses_per_epoch = [[] for _ in range(args.k_fold)]
            training_acc_per_epoch = [[] for _ in range(args.k_fold)]
            validation_losses_per_epoch = [[] for _ in range(args.k_fold)]
            validation_acc_per_epoch = [[] for _ in range(args.k_fold)]

            n_iter_step.append(1)

            kf: KFold = KFold(n_splits=args.k_fold, shuffle=True)

            all_scores = []
            k_step = -1
            for train_index, val_index in kf.split(img_train):
                train_images, val_images = img_train[train_index], img_train[val_index]
                train_labels, val_labels = label_train[train_index], label_train[val_index]
                k_step += 1

                history, strategy, time_history, train_metrics, interrupt = train_model(train_images, train_labels,
                                                                                        val_images,
                                                                                        val_labels,
                                                                                        params["lambda_l2"],
                                                                                        input_shape, num_classes,
                                                                                        strategy_params,
                                                                                        args)

                training_acc_per_epoch[k_step].append(history['accuracy'])
                validation_acc_per_epoch[k_step].append(history['val_accuracy'])
                training_losses_per_epoch[k_step].append(history['loss'])
                validation_losses_per_epoch[k_step].append(history['val_loss'])
                train_metrics_list[0] = train_metrics
                all_scores.append(np.max(history['val_accuracy']))
                # all_scores.append(np.min(history['val_loss']))

            print("Mean val accuracy:", np.mean(all_scores),
                  "Time taken: {:.2f}".format(time.time() - search_step_start_time))

            train_acc.append(training_acc_per_epoch)
            val_acc.append(validation_acc_per_epoch)
            train_loss.append(training_losses_per_epoch)
            val_loss.append(validation_losses_per_epoch)

            return -np.mean(all_scores)
            # return np.mean(all_scores)

        result = gp_minimize(objective, search_space, n_calls=args.n_calls,  # acq_func='EI',
                             # x0=[[1e-6], [1e-4], [1e-2]],
                             # n_random_starts=3,
                             # n_jobs=3,
                             random_state=45
                             )

        print("Best lambda: {}".format(result.x))
        print("Best validation loss: {}".format(result.fun))
        metrics = {
            "training_loss": train_loss,
            "training_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": args,
            "setup": train_metrics_list[0]
        }
        result["metrics"] = metrics
        dump(result,
             '../results/bayesian/bayesian_result_{}_{}_{}.pkl'.format(strategy.get_file_name(), args.model.lower(),
                                                                       datetime.now().strftime(
                                                                           '%m_%d_%H_%M_%S')),
             store_objective=False)
        print(f"Finished search for {strategy.get_plot_title()}")

    else:
        print("==> Starting Training")
        if args.train_on_baseline == 1:
            lambda_l2 = get_l2_lambda(args, **{"optimizer": "sgd", "compression": "none"})
        elif args.train_on_baseline == 2:
            lambda_l2 = get_l2_lambda(args, **strategy_params)
        else:
            lambda_l2 = None  # 5e-4
        args.lambda_l2 = lambda_l2
        print("Using L2 lambda:", lambda_l2)
        if args.k_fold > 1:
            kf = KFold(n_splits=args.k_fold, shuffle=True)

            training_losses_per_epoch = [[] for _ in range(args.k_fold)]
            training_acc_per_epoch = [[] for _ in range(args.k_fold)]
            validation_losses_per_epoch = [[] for _ in range(args.k_fold)]
            validation_acc_per_epoch = [[] for _ in range(args.k_fold)]
            compression_rates = [[] for _ in range(args.k_fold)]

            k_step = -1
            for train_index, val_index in kf.split(img_train):
                train_images, val_images = img_train[train_index], img_train[val_index]
                train_labels, val_labels = label_train[train_index], label_train[val_index]
                k_step += 1

                history, strategy, time_history, train_metrics, interrupt = train_model(train_images, train_labels,
                                                                                        val_images,
                                                                                        val_labels,
                                                                                        lambda_l2,
                                                                                        input_shape, num_classes,
                                                                                        strategy_params,
                                                                                        args)

                training_acc_per_epoch[k_step] = history['accuracy']
                validation_acc_per_epoch[k_step] = history['val_accuracy']
                training_losses_per_epoch[k_step] = history['loss']
                validation_losses_per_epoch[k_step] = history['val_loss']
                compression_rates[k_step].append(strategy.compression.compression_rates)
        else:
            history, strategy, time_history, train_metrics, interrupt = train_model(img_train, label_train, img_test,
                                                                                    label_test,
                                                                                    lambda_l2,
                                                                                    input_shape, num_classes,
                                                                                    strategy_params,
                                                                                    args)

            training_acc_per_epoch = history['accuracy']
            validation_acc_per_epoch = history['val_accuracy']
            training_losses_per_epoch = history['loss']
            validation_losses_per_epoch = history['val_loss']
            if strategy.compression is not None:
                compression_rates = [np.mean(strategy.compression.compression_rates)]
            elif strategy.optimizer_name != "sgd":
                compression_rates = [np.mean(strategy.optimizer.compression_rates)]
            else:
                compression_rates = [1]

        metrics = {
            "training_loss": str(training_losses_per_epoch),
            "training_acc": str(training_acc_per_epoch),
            "val_loss": str(validation_losses_per_epoch),
            "val_acc": str(validation_acc_per_epoch),
            "args": vars(args),
            "compression_rates": compression_rates,
            "setup": train_metrics,
            "time_per_epoch": time_history  # .epoch_times,
        }
        if interrupt:
            file = open('../results/compression/interrupted_training_{}_{}_'
                        '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                         datetime.now().strftime('%m_%d_%H_%M_%S')),
                        "w")
        else:
            file = open('../results/compression/training_{}_{}_'
                        '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                         datetime.now().strftime('%m_%d_%H_%M_%S')),
                        "w")
        json.dump(metrics, file, indent=4)
        file.close()
        print(f"Finished training for {strategy.get_plot_title()}")

        if interrupt:
            raise KeyboardInterrupt


def main():
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()
