import argparse
import os
import time
from datetime import datetime
import json
from pprint import pprint

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from models.LeNet import LeNet
from models.ResNet import ResNet
from models.MobileNet import MobileNet
from models.DenseNet import DenseNet
from compressions.bSGD import bSGD
from models.VGG import VGG
from models.MobileNetV2 import MobileNetV2
from utilities import Strategy

from utilities.custom_callbacks import TimeHistory, CosineDecayCallback, step_decay
from utilities.datasets import load_dataset

from compressions.TernGrad import TernGrad
from compressions.NaturalCompression import NaturalCompression

from compressions.GradientSparsification import GradientSparsification
from compressions.OneBitSGD import OneBitSGD
from compressions.SparseGradient import SparseGradient
from compressions.Atomo import Atomo
from compressions.TopK import TopK
from compressions.vqSGD import vqSGD


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--strategy', type=str, help='Strategy')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--n_calls', type=int, help='Bayesian Search iterations')
    parser.add_argument('--stop_patience', type=int, help='Early stopping patience')
    parser.add_argument('--lr_decay', type=int, help='Lr decay')
    parser.add_argument('--k_fold', type=int, help='K-Fold')
    parser.add_argument('--lambda_l2', type=float, help='L2 regularization lambda')
    parser.add_argument('--fullset', type=float, help='% of dataset')
    parser.add_argument('--log', type=int, help='Log')
    parser.add_argument('--gpu', type=int, help='GPU')
    parser.add_argument('--train_on_baseline', type=int, help='Take baseline L2')
    parser.add_argument('--bayesian_search', action='store_true', help='Apply Bayesian search')
    return parser.parse_args()


def model_factory(model_name, lambda_l2, input_shape, num_classes):
    print(f"Initializing {model_name.upper()}")
    if model_name == "lenet":
        model = LeNet(search=True).search_model(lambda_l2=lambda_l2)
        return model
    elif model_name == "resnet18":
        model = ResNet("resnet18", num_classes, lambda_l2=lambda_l2)
        return model
    elif model_name == "mobilenet":
        model = MobileNetV2(num_classes, lambda_l2=lambda_l2)
        return model
    elif model_name == "vgg11":
        model = VGG(vgg_name="vgg11", num_classes=num_classes, lambda_l2=lambda_l2)
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
        # not working yet
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=Atomo(svd_rank=params["svd_rank"]))
    elif params["compression"].lower() == "none":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=None, optimizer=params["optimizer"].lower())


def get_l2_lambda(args, **params) -> float:
    lambdas = None
    if args.model.lower() == "lenet":
        lambdas = json.load(open("../results/lambda_lookup.json", "r"))
    elif args.model.lower() == "resnet18":
        lambdas = json.load(open("../results/lambda_lookup_resnet18.json", "r"))
    elif args.model.lower() == "vgg11":
        lambdas = json.load(open("../results/lambda_lookup_vgg11.json", "r"))

    opt = params["optimizer"]
    comp = params["compression"]

    if (opt in ["efsignsgd", "sgd"] and comp == "none") or comp in ["onebitsgd", "naturalcompression"]:
        return lambdas[opt][comp]

    keys = [k for k in params.keys() if k != "compression" and k != "optimizer" and k != "learning_rate"]
    first_key = list(lambdas[opt][comp].keys())[0]
    first_key_value = lambdas[opt][comp][first_key][str(params[first_key])]

    if type(first_key_value) != float:
        second_key = list(first_key_value.keys())[0]
        print(second_key, first_key_value)
        if second_key in keys:
            second_key_value = first_key_value[second_key][str(params[second_key])]
            return second_key_value
    return first_key_value


def train_model(train_images, train_labels, val_images, val_labels, lambda_l2, input_shape, num_classes,
                strategy_params, args):
    model = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)
    strategy = strategy_factory(**strategy_params)
    strategy.summary()

    BATCH_SIZE = 32
    initial_lr = strategy_params["learning_rate"]
    drop_factor = 0.5
    drop_epochs = [25, 37]
    min_lr = initial_lr * 0.1 * 0.1

    if args.dataset == "cifar10" and args.model.lower() == "resnet18":
        BATCH_SIZE = 256
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.1
        drop_epochs = [15, 30]
        min_lr = initial_lr * 0.1 * 0.1

    elif args.dataset == "cifar10" and args.model.lower() == "vgg11":
        BATCH_SIZE = 128
        initial_lr = strategy_params["learning_rate"]
        drop_factor = 0.2
        drop_epochs = [15, 30]
        min_lr = initial_lr * 0.1 * 0.1

    print("BATCH SIZE:", BATCH_SIZE)
    time_callback = TimeHistory()
    callbacks = [time_callback]

    if args.stop_patience < args.epochs:
        early_stopping = EarlyStopping(monitor='val_loss', patience=args.stop_patience, verbose=1)
        callbacks.append(early_stopping)

        lr_scheduler = LearningRateScheduler(lambda epoch: step_decay(epoch, initial_lr, drop_factor, drop_epochs,
                                                                      min_lr))
        callbacks.append(lr_scheduler)

    if args.bayesian_search:
        model.compile(optimizer=strategy,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=strategy,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=args.lr_decay, cooldown=2 * args.lr_decay,
                                      min_lr=1e-04)

        lr_scheduler = LearningRateScheduler(
            lambda epoch: step_decay(epoch, initial_lr, drop_factor, drop_epochs, min_lr))

        callbacks.append(lr_scheduler)

    history = model.fit(train_images, train_labels, epochs=args.epochs,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_images, val_labels), verbose=args.log, callbacks=callbacks)

    train_metrics = {"batch_size": BATCH_SIZE, "lr_decay": drop_epochs, "drop_factor": drop_factor, "min_lr": min_lr}
    return history, strategy, time_callback, train_metrics


def worker(args):
    if args.gpu != 1:
        print("Using CPU")
        tf.config.set_visible_devices([], 'GPU')
        print(tf.config.get_visible_devices())
    else:
        print("Using GPU")
    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset(args.dataset,
                                                                                          fullset=args.fullset)
    strategy_params = json.loads(args.strategy)
    strategy = strategy_factory(**strategy_params)
    print(strategy_params)

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
        print(strategy.get_file_name())

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

                history, strategy, time_history, train_metrics = train_model(train_images, train_labels, val_images,
                                                                             val_labels,
                                                                             params["lambda_l2"],
                                                                             input_shape, num_classes, strategy_params,
                                                                             args)

                training_acc_per_epoch[k_step].append(history.history['accuracy'])
                validation_acc_per_epoch[k_step].append(history.history['val_accuracy'])
                training_losses_per_epoch[k_step].append(history.history['loss'])
                validation_losses_per_epoch[k_step].append(history.history['val_loss'])
                train_metrics_list[0] = train_metrics

                all_scores.append(np.max(history.history['val_accuracy']))
                # all_scores.append(np.mean(history.history['val_loss']))

            print("Mean val accuracy:", np.mean(all_scores),
                  "Time taken: {:.2f}".format(time.time() - search_step_start_time))

            train_acc.append(training_acc_per_epoch)
            val_acc.append(validation_acc_per_epoch)
            train_loss.append(training_losses_per_epoch)
            val_loss.append(validation_losses_per_epoch)

            return -np.mean(all_scores)

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
        print("--- Training ---")
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

                history, strategy, time_history, train_metrics = train_model(train_images, train_labels, val_images,
                                                                             val_labels,
                                                                             lambda_l2,
                                                                             input_shape, num_classes, strategy_params,
                                                                             args)

                training_acc_per_epoch[k_step] = history.history['accuracy']
                validation_acc_per_epoch[k_step] = history.history['val_accuracy']
                training_losses_per_epoch[k_step] = history.history['loss']
                validation_losses_per_epoch[k_step] = history.history['val_loss']
                compression_rates[k_step].append(strategy.compression.compression_rates)
        else:
            compression_rates = []

            history, strategy, time_history, train_metrics = train_model(img_train, label_train, img_test, label_test,
                                                                         lambda_l2,
                                                                         input_shape, num_classes, strategy_params,
                                                                         args)

            training_acc_per_epoch = history.history['accuracy']
            validation_acc_per_epoch = history.history['val_accuracy']
            training_losses_per_epoch = history.history['loss']
            validation_losses_per_epoch = history.history['val_loss']
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
            "time_per_epoch": time_history.epoch_times,
            "time_per_step": np.mean(time_history.times),

        }
        file = open('../results/compression/training_{}_{}_'
                    '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                     datetime.now().strftime('%m_%d_%H_%M_%S')),
                    "w")
        json.dump(metrics, file, indent=4)
        file.close()
        print(f"Finished training for {strategy.get_plot_title()}")


def main():
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()
