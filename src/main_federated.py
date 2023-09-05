import json
from datetime import datetime
import os

from optimizer.FetchSGD import FetchSGD
from main_local import strategy_factory, model_factory
from utilities.federator import *
from utilities.parameters import get_parameters_federated
from models.LeNet import LeNet
from compressions.bSGD import bSGD
from models.ResNet import ResNet
from utilities.datasets import load_dataset
from utilities.client_data import client_datasets, stratified_sampling, label_splitter, plot_client_distribution, \
    client_datasets_cifar
from utilities import Strategy
from compressions.TernGrad import TernGrad

# initialize GPU usage
# Restrict TensorFlow to only allocate 2GB of memory on GPUs
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
"""


def fed_worker(args):
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
    strategy.summary()

    lambda_l2 = 0.0015
    model_client = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)

    print(strategy_params)

    start_time = time.time()
    # VARIABLES
    max_iter = args.max_iter
    number_clients = args.number_clients
    inactive_prob = 0
    batch_size = args.batch_size
    early_stopping = args.es
    early_stopping_rate = args.es_rate
    varying_local_iteration = args.varying_local_iter
    # set random seed
    np.random.seed(args.rs)

    # number of local iterations
    local_iter_type = args.local_iter_type
    if local_iter_type == 'same':
        local_iter = [args.const_local_iter for i in range(number_clients)]
    elif local_iter_type == 'uniform':
        mean = 3
        min_value = mean - 2
        max_value = mean + 2
        local_iter = np.random.choice(range(min_value, max_value + 1), number_clients)
    elif local_iter_type == 'gaussian':
        if varying_local_iteration:
            mean = np.round(np.random.dirichlet([0.5] * number_clients) * 30)
            std = 1
            local_iter = [np.maximum(np.round(np.random.normal(mean[i], std)), 1) for i in range(number_clients)]
        else:
            mean = 3
            std = 1
            local_iter = np.maximum(np.round(np.random.normal(mean, std, number_clients)), 1)
    elif local_iter_type == 'exponential':
        if varying_local_iteration:
            mean = np.round(np.random.dirichlet([0.5] * number_clients) * 30)
            local_iter = [np.maximum(np.round(np.random.exponential(mean[i])), 1) for i in range(number_clients)]
        else:
            mean = 3
            local_iter = np.maximum(np.round(np.random.exponential(mean, number_clients)), 1)
    elif local_iter_type == 'dirichlet':
        local_iter = np.round(np.random.dirichlet([0.5] * number_clients) * 30)
        # baseline to compare with when the local iteration is varying
    else:
        print('Not defined local iteration type!')

    learning_rate = args.learning_rate
    num_sample = 100
    split_type = args.split_type

    loss_func = SparseCategoricalCrossentropy()
    acc_func = SparseCategoricalAccuracy()
    loss_metric = keras.metrics.SparseCategoricalCrossentropy()

    (split_data, split_labels) = label_splitter(img_train, label_train)

    # stratified sampling from training data
    val_sample_per_class = 100
    (validation_images, validation_labels) = stratified_sampling(num_sample_per_class=val_sample_per_class,
                                                                 list_data=split_data, list_labels=split_labels)

    (client_data, client_labels) = client_datasets_cifar(number_clients, img_train, label_train, num_classes,
                                                         shuffle=True)

    # (client_data, client_labels) = client_datasets(number_clients=number_clients, split_type="random",
    #                                                list_data=split_data, list_labels=split_labels, beta=0.05)

    # Plot Client Class Distribution
    # plot_client_distribution(number_clients, client_labels)
    def count_labels_per_client(client_labels, num_classes):
        label_counts_per_client = []

        for labels in client_labels:
            counts = np.bincount(labels.flatten(), minlength=num_classes)
            label_counts_per_client.append(counts)

        return label_counts_per_client

    # Given the client_datasets_cifar function and the above count_labels_per_client function
    (client_data, client_labels) = client_datasets_cifar(number_clients, img_train, label_train, num_classes,
                                                         shuffle=True)
    label_counts = count_labels_per_client(client_labels, num_classes)

    # Print the counts for each client
    for i, counts in enumerate(label_counts):
        print(f"Client {i}: {counts}")

    time_pre = time.time() - start_time

    # Federated Learning
    active_client_matrix = active_client(prob=inactive_prob, max_iter=max_iter, number_clients=number_clients)

    if args.dataset == "cifar10":
        model_client.build(input_shape=(None, 32, 32, 3))

    model_client.compile(optimizer=strategy,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    model_client.summary()

    time_create_model = time.time() - start_time - time_pre

    model_federated, test_loss, test_acc, compression_rates, time_per_iteration = federator(
        active_clients=active_client_matrix,
        learning_rate=learning_rate,
        model=model_client,
        train_data=client_data,
        train_label=client_labels, test_data=img_test,
        test_label=label_test,
        number_clients=number_clients,
        max_iter=max_iter, batch_size=batch_size,
        local_iter=local_iter,
        val_data=validation_images,
        val_label=validation_labels,
        loss_func=loss_func, acc_func=acc_func,
        optimizer=strategy,
        loss_metric=loss_metric,
        num_sample=num_sample, num_class=num_classes,
        early_stopping=early_stopping,
        es_rate=early_stopping_rate,
        varying_local_iter=varying_local_iteration,
        folder_name="folder_name",
        local_iter_type=local_iter_type)

    time_federator = time.time() - time_pre - time_create_model - start_time

    fed_metrics = {
        "test_loss": str(test_loss),
        "test_acc": str(test_acc),
        "args": {k: v for k, v in vars(args).items() if v is not None}
        ,
        "compression_rates": compression_rates,
        "times": {
            "create_model": time_create_model,
            "federator": time_federator
        },
        "time_per_iteration": time_per_iteration
    }
    print(
        f'time split data: {time_create_model:.4f} \ntime create model: {time_create_model:.4f} \ntime federator: {time_federator:.4f}')

    file = open('../results/federated/{}_{}_'
                '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                 datetime.now().strftime('%m_%d_%H_%M_%S')),
                "w")
    json.dump(fed_metrics, file, indent=4)
    file.close()
    print(f"Finished federated training for {strategy.get_plot_title()}")


def main():
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = get_parameters_federated()

    fed_worker(args)


if __name__ == "__main__":
    main()
