import argparse


def parse_args():
    """Local Parameter"""

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


def get_parameters_federated():
    """Federated Parameter"""
    parser = argparse.ArgumentParser()

    """Strategy"""
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--strategy', type=str, help='Strategy')
    parser.add_argument('--stop_patience', type=int, help='Early stopping patience')
    parser.add_argument('--fullset', type=float, help='% of dataset', default=100)
    parser.add_argument('--log', type=int, help='Log')
    parser.add_argument('--gpu', type=int, help='GPU')
    parser.add_argument('--train_on_baseline', type=int, help='Take baseline L2')

    """Federated"""
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--number_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)  # 500
    parser.add_argument('--rs', type=int, default=100)  # random seed
    parser.add_argument('--local_iter_type', type=str, default='same',
                        help='Choose from same, uniform, gaussian, exponential and dirichlet(baseline to compare with when number of local iterations changes.).')
    parser.add_argument('--const_local_iter', type=int,
                        default=1)  # the number of local iterations when the number of local iterations is constant
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--varying_local_iter', action='store_true', default=False)

    return parser.parse_args()


def get_parameters():
    """Federated Parameter Old"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--number_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)  # 500
    parser.add_argument('--es', action='store_false', default=True)  # early stopping
    parser.add_argument('--es_rate', type=float, default=1e-3)
    parser.add_argument('--varying_local_iter', action='store_true', default=False)
    parser.add_argument('--rs', type=int, default=100)  # random seed
    parser.add_argument('--local_iter_type', type=str, default='exponential',
                        help='Choose from same, uniform, gaussian, exponential and dirichlet(baseline to compare with when number of local iterations changes.).')
    parser.add_argument('--const_local_iter', type=int,
                        default=1)  # the number of local iterations when the number of local iterations is constant
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--split_type', type=str, default='dirichlet', help='Choose from dirichlet, uniform')
    parser.add_argument('--beta_index', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--beta_mode', type=str, default='value', help='choose beta mode from index and value')
    parser.add_argument('--experiment', type=int, default=0,
                        help='choose experiment number from 1, 2, 3, 4, 5. more information can be found in experiment.py. default 0, logs will be stored in a folder named with current time.')

    param = parser.parse_args()

    return param


def save_parameters(parameters, file_path):
    with open(file_path, 'w') as file:
        for param, value in vars(parameters).items():
            file.write(f"{param}: {value}\n")
    return
