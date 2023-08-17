import argparse
import time
from datetime import datetime
import json
from pprint import pprint

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import dump, gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

#from models.ResNet import ResNet
from models.LeNet import LeNet
from models.ResNet import ResNet18_new

from utilities.datasets import load_dataset

from compressions.TernGrad import TernGrad
from compressions.NaturalCompression import NaturalCompression

from compressions.GradientSparsification import GradientSparsification
from compressions.OneBitSGD import OneBitSGD
from compressions.SparseGradient import SparseGradient
from compressions.Atomo import Atomo
from compressions.TopK import TopK
from compressions.vqSGD import vqSGD

from strategy import Strategy


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
    if model_name == "lenet":
        return LeNet(search=True).search_model(lambda_l2)
    elif model_name == "resnet":
        model = ResNet18_new(num_classes=num_classes, lambda_l2=lambda_l2) # ResNet().search_model(lambda_l2, input_shape, num_classes)
        model.build(input_shape=(None, 32, 32, 3))
        return model
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def strategy_factory(**params) -> Strategy:
    if params["compression"].lower() == "terngrad":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=TernGrad(params["clip"]))
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
                        compression=Atomo(sparsity_budget=params["sparsity_budget"]))
    elif params["compression"].lower() == "none":
        return Strategy(learning_rate=params["learning_rate"], params=params,
                        compression=None, optimizer=params["optimizer"].lower())


def get_l2_lambda(**params) -> float:
    lambdas = json.load(open("results/lambda_lookup.json", "r"))
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


def worker(args):
    if args.gpu != 1:
        tf.config.set_visible_devices([], 'GPU')
    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset(args.dataset,
                                                                                          fullset=args.fullset)
    strategy_params = json.loads(args.strategy)
    strategy = strategy_factory(**strategy_params)
    print(input_shape, num_classes)

    if args.bayesian_search:
        print("Bayesian Search")
        search_space = [Real(1e-5, 1e0, "log-uniform", name='lambda_l2')]
        train_loss = []
        val_loss = []
        val_acc = []
        train_acc = []

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

                model = model_factory(args.model.lower(), params["lambda_l2"], input_shape, num_classes)
                strategy = strategy_factory(**strategy_params)

                model.compile(optimizer=strategy,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=args.stop_patience, verbose=1)
                # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                #                               patience=max(5, args.stop_patience - 5),
                #                               min_lr=strategy_params["learning_rate"]/20)

                history = model.fit(train_images, train_labels, epochs=args.epochs,
                                    batch_size=32,
                                    validation_data=(val_images, val_labels), verbose=args.log,
                                    callbacks=[early_stopping])

                training_acc_per_epoch[k_step].append(history.history['accuracy'])
                validation_acc_per_epoch[k_step].append(history.history['val_accuracy'])
                training_losses_per_epoch[k_step].append(history.history['loss'])
                validation_losses_per_epoch[k_step].append(history.history['val_loss'])

                all_scores.append(np.mean(history.history['val_accuracy']))

            print("   Mean val accuracy:", np.mean(all_scores),
                  "Time taken: {:.2f}".format(time.time() - search_step_start_time))

            train_acc.append(training_acc_per_epoch)
            val_acc.append(validation_acc_per_epoch)
            train_loss.append(training_losses_per_epoch)
            val_loss.append(validation_losses_per_epoch)
            return - np.mean(all_scores)

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
        result = gp_minimize(objective, search_space, n_calls=args.n_calls, #acq_func='EI',
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
            "args": args
        }
        result["metrics"] = metrics
        dump(result, f'results/bayesian/bayesian_result_{strategy.get_file_name()}_{args.dataset}.pkl',
             store_objective=False)
        print(f"Finished search for {strategy.get_plot_title()}")

    else:
        print("Training")
        if args.train_on_baseline == 1:
            lambda_l2 = get_l2_lambda(**{"optimizer": "sgd", "compression": "none"})
        elif args.train_on_baseline == 2:
            lambda_l2 = get_l2_lambda(**strategy_params)
        else:
            lambda_l2 = 0
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

                model = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)

                strategy = strategy_factory(**strategy_params)
                strategy.summary()

                model.compile(optimizer=strategy,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                callbacks = []
                if args.stop_patience < args.epochs:
                    early_stopping = EarlyStopping(monitor='val_loss', patience=args.stop_patience, verbose=1)
                    callbacks = [early_stopping]

                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=args.lr_decay,
                                              min_lr=strategy_params["learning_rate"] / 20)
                callbacks.append(reduce_lr)

                history = model.fit(train_images, train_labels, epochs=args.epochs,
                                    batch_size=32,
                                    validation_data=(val_images, val_labels), verbose=args.log, callbacks=callbacks)

                training_acc_per_epoch[k_step] = history.history['accuracy']
                validation_acc_per_epoch[k_step] = history.history['val_accuracy']
                training_losses_per_epoch[k_step] = history.history['loss']
                validation_losses_per_epoch[k_step] = history.history['val_loss']
                compression_rates[k_step].append(strategy.compression.compression_rates)
        else:
            compression_rates = []

            model = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)

            strategy = strategy_factory(**strategy_params)
            strategy.summary()

            model.compile(optimizer=strategy,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            callbacks = []
            if args.stop_patience < args.epochs:
                early_stopping = EarlyStopping(monitor='val_loss', patience=args.stop_patience, verbose=1)
                callbacks = [early_stopping]

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=args.lr_decay,
                                          min_lr=strategy_params["learning_rate"] / 20)
            callbacks.append(reduce_lr)

            BATCH_SIZE = 32
            if args.dataset == "cifar10":
                BATCH_SIZE = 64

            history = model.fit(img_train, label_train, epochs=args.epochs,
                                batch_size=BATCH_SIZE,
                                validation_data=(img_test, label_test), verbose=args.log, callbacks=callbacks)

            training_acc_per_epoch = history.history['accuracy']
            validation_acc_per_epoch = history.history['val_accuracy']
            training_losses_per_epoch = history.history['loss']
            validation_losses_per_epoch = history.history['val_loss']
            if strategy.compression is not None:
                compression_rates = [np.mean(strategy.compression.compression_rates)]
            elif strategy.optimizer_name != "sgd":
                compression_rates = [np.mean(strategy.optimizer.compression_rates)]

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
        print(f"Finished training for {strategy.get_plot_title()}")


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
    # print(get_l2_lambda(**{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 2}))
    # print(get_l2_lambda(**{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 1}))
