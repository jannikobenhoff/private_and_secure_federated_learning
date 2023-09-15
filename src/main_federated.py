import json
from datetime import datetime
import os

from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args, dump

from main_local import strategy_factory, model_factory, get_l2_lambda
from utilities.federator import *
from utilities.parameters import get_parameters_federated
from models.ResNet import resnet
from utilities.datasets import load_dataset
from utilities.client_data import client_datasets, stratified_sampling, label_splitter  # plot_client_distribution, \

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

    # (img_train, label_train), (img_test, label_test) = keras.datasets.cifar10.load_data()
    # img_train, img_test = img_train / 255.0, img_test / 255.0

    # Initialize Strategy (Optimizer / Compression)
    learning_rate = args.learning_rate
    strategy_params = json.loads(args.strategy)
    strategy_params["learning_rate"] = learning_rate
    strategy = strategy_factory(**strategy_params)
    strategy.summary()
    print(strategy_params)

    # Initialize Model and build if needed
    if args.bayesian_search:
        lambda_l2 = args.search_lambda
    elif True:  # args.train_on_baseline == 1:
        lambda_l2 = None  # get_l2_lambda(args, fed=True, **{"optimizer": "sgd", "compression": "none"})
    else:
        lambda_l2 = None
    # lambda_l2 = None

    args.lambda_l2 = lambda_l2
    print("Using L2 lambda:", lambda_l2)
    model_client = model_factory(args.model.lower(), lambda_l2, input_shape, num_classes)

    # if args.dataset == "cifar10":
    #     model_client.build(input_shape=(None, 32, 32, 3))

    model_client.compile(optimizer=strategy.optimizer_update,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    # model_client.summary()

    start_time = time.time()
    # VARIABLES
    max_iter = args.max_iter
    number_clients = args.number_clients
    inactive_prob = 0
    batch_size = args.batch_size
    varying_local_iteration = args.varying_local_iter
    beta = args.beta
    split_type = args.split_type
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

    loss_func = SparseCategoricalCrossentropy()
    acc_func = SparseCategoricalAccuracy()
    loss_metric = tf.keras.metrics.Mean()

    # Plot Client Class Distribution
    # plot_client_distribution(number_clients, client_labels)

    time_pre = time.time() - start_time

    # Federated Learning
    active_client_matrix = active_client(prob=inactive_prob, max_iter=max_iter, number_clients=number_clients)

    time_create_model = time.time() - start_time - time_pre
    (split_data, split_labels) = label_splitter(img_train, label_train)
    num_classes = len(split_labels)

    # stratified sampling from training data
    val_sample_per_class = 100
    (validation_images, validation_labels) = stratified_sampling(num_sample_per_class=val_sample_per_class,
                                                                 list_data=split_data, list_labels=split_labels)

    (client_data, client_labels) = client_datasets(number_clients=number_clients, split_type=split_type,
                                                   list_data=split_data, list_labels=split_labels, beta=beta,
                                                   dataset=args.dataset)

    def count_labels_per_client(client_labels, num_classes):
        label_counts_per_client = []

        for labels in client_labels:
            counts = np.bincount(labels.flatten(), minlength=num_classes)
            label_counts_per_client.append(counts)

        return label_counts_per_client

    label_counts = count_labels_per_client(client_labels, num_classes)

    for i, counts in enumerate(label_counts):
        print(f"Client {i}: {counts}")

    # Start FL
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
        loss_metric=loss_metric, num_class=num_classes,
        varying_local_iter=varying_local_iteration,
        folder_name="folder_name",
        local_iter_type=local_iter_type)

    time_federator = time.time() - time_pre - time_create_model - start_time

    # Save FL metrics
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
        "client_labels": str(label_counts),
        "time_per_iteration": time_per_iteration
    }
    print(
        f'time split data: {time_create_model:.4f} \ntime create model: {time_create_model:.4f} \ntime federator: {time_federator:.4f}')

    if args.bayesian_search:
        file = open('../results/federated_bayesian/{}_{}_'
                    '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                     datetime.now().strftime('%m_%d_%H_%M_%S')),
                    "w")
    else:
        file = open('../results/federated/{}_{}_'
                    '{}.json'.format(strategy.get_file_name(), args.model.lower(),
                                     datetime.now().strftime('%m_%d_%H_%M_%S')),
                    "w")
    json.dump(fed_metrics, file, indent=4)
    file.close()
    print(f"Finished federated training for {strategy.get_plot_title()}")
    return np.max(test_acc)


def bayesian_search(args):
    space = [Real(1e-6, 1e-1, "log-uniform", name='l2_reg')]
    iteration_dummy = []

    @use_named_args(space)
    def objective(**params):
        # params["l2_reg"] = params["l2_reg"]
        print("Lambda:", params["l2_reg"])
        print("Iteration:", len(iteration_dummy) + 1)
        all_scores = []

        for i in range(3):
            print(f"Step {i + 1}/3")
            args.search_lambda = params["l2_reg"]
            max_test_acc = fed_worker(args)
            all_scores.append(max_test_acc)

        iteration_dummy.append(1)
        return - np.mean(all_scores)

    result = gp_minimize(objective, space, n_calls=10, verbose=0, random_state=45)

    metrics = {
        "args": args,

    }
    result["metrics"] = metrics
    dump(result,
         '../results/bayesian/bayesian_result_{}_{}.pkl'.format(args.model.lower(),
                                                                datetime.now().strftime(
                                                                    '%m_%d_%H_%M_%S')),
         store_objective=False)
    print(f"Finished Bayesian Search.")


def main():
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = get_parameters_federated()

    if args.bayesian_search:
        bayesian_search(args)
    else:
        fed_worker(args)


if __name__ == "__main__":
    main()
