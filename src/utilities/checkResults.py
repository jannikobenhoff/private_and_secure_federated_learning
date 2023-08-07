import os
from tensorboard import program
from skopt import load, dump

tracking_address = '../results/logs/scalars/baseline_model-20230718-181613'


def call_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', os.path.abspath(tracking_address)])
    tb.main()


def change_files(result_file, metric_file):
    result = load("../results/bayesian/" + result_file + ".pkl")
    metric = load("../results/bayesian/" + metric_file + ".pkl")
    result["metrics"] = metric
    dump(result, "../results/bayesian/" + result_file + ".pkl", store_objective=False)


if __name__ == "__main__":
    change_files("bayesian_result_EFsignSGD_0108", "bayesian_metrics_EFsignSGD_0108")
