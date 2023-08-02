import numpy as np
from tensorflow import Tensor
import tensorflow as tf


class Strategy:
    def __init__(self, optimizer, compression = None):
        self.optimizer = optimizer
        self.compression = compression
        self.compression_ratio = []
        self.iter = 0

    def update_parameters(self, grads_and_vars: zip):
        self.compression_ratio.append([])
        grads_and_vars = list(grads_and_vars)
        gradient, variables = zip(*grads_and_vars)
        gradient = list(gradient)

        if self.compression is not None:
            scope_name = "optimizer"
            with tf.name_scope(scope_name):
                with tf.init_scope():
                    # Lift variable creation to init scope to avoid environment
                    # issues.
                    self.compression.build(variables)
        if self.compression is None:
            self.optimizer.apply_gradients(zip(gradient, variables))
        else:
            gradient_compressed = []
            for i, grad in enumerate(gradient):
                gradient_compressed.append(self.compression.compress(grad, variables[i]))
                # self.compression_ratio[self.iter].append(get_compression_rate(gradient[i], gradient_compressed[i]))
            # print("compression ratio:", (get_compression_rate(gradient[1], gradient_compressed[1])))
            # count_tensor_values(gradient_compressed[0])
            self.optimizer.apply_gradients(zip(gradient_compressed, variables))

        self.iter += 1

    def summary(self, add: str = ""):
        if self.compression is None:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: None\n--- {add}")
        else:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: {self.compression.name}\n--- {add}")

    def get_plot_title(self):
        if self.compression is None:
            return "{} - {:.4f}".format(self.optimizer.name,
                                        self.optimizer.learning_rate.numpy())
        else:
            return "{} - {} - {:.4f}".format(self.optimizer.name,
                                             self.compression.name,
                                             self.optimizer.learning_rate.numpy())

    def get_file_name(self):
        if self.compression is None:
            return "{}".format(self.optimizer.name)
        else:
            return "{}_{}".format(self.optimizer.name,
                                             self.compression.name)

