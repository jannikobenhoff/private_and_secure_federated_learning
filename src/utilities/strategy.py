import numpy as np
from tensorflow import Tensor
import tensorflow as tf
from src.compressions.Compression import Compression
from src.utilities.compression_rate import get_compression_rate


class Strategy:
    def __init__(self, optimizer, compression: Compression = None):
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
                # self.compression_ratio[self.iter].append(get_compression_rate(gradient_uncompressed[i], gradient[i]))
            # print("compression ratio:", (get_compression_rate(gradient[1], gradient_compressed[1])))

            self.optimizer.apply_gradients(zip(gradient_compressed, variables))

        self.iter += 1

    def summary(self):
        if self.compression is None:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: None\n---")
        else:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: {self.compression.name}\n---")

    def get_plot_title(self):
        return "{} - {} - {:.4f}".format(self.optimizer.name,
                                         self.compression.name,
                                         self.optimizer.learning_rate.numpy())
