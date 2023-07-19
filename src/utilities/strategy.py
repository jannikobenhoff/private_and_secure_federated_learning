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
            gradient_uncompressed = gradient
            for i, grad in enumerate(gradient):
                gradient[i] = self.compression.compress(grad, variables[i])
                self.compression_ratio[self.iter].append(get_compression_rate(gradient_uncompressed[i], gradient[i]))
            self.optimizer.apply_gradients(zip(gradient, variables))
        # print("Avg. compression ratio:", sum(self.compression_ratio[self.iter])/len(self.compression_ratio[self.iter]))

        self.iter += 1

    def summary(self):
        if self.compression is None:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: None\n---")
        else:
            print(f"---\nOptimizer: {self.optimizer.name}\nCompression: {self.compression.name}\n---")


