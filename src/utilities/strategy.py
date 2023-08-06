import numpy as np
from tensorflow import Tensor
import tensorflow as tf

# from ..compressions.TernGrad import TernGrad
# from ..compressions.NaturalCompression import NaturalCompression
# from ..optimizer.EFsignSGD import EFsignSGD
# from ..optimizer.FetchSGD import FetchSGD
# from ..optimizer.MemSGD import MemSGD
# from ..compressions.GradientSparsification import GradientSparsification
# from ..compressions.OneBitSGD import OneBitSGD
# from ..compressions.SparseGradient import SparseGradient
# from ..compressions.Atomo import Atomo
# from ..compressions.TopK import TopK
# from ..compressions.vqSGD import vqSGD
#from ..optimizer.SGD import SGD

# from .utilities.huffman import decode_rle, decode_huffman


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
                if False:  # huffman
                    enc, huf, shape = self.compression.compress(grad, variables[i])
                    dec_huf = decode_huffman(enc, huf)
                    dec = decode_rle(dec_huf)
                    gradient_compressed.append(tf.reshape(dec, shape))
                else:
                    gradient_compressed.append(self.compression.compress(grad, variables[i]))

            self.optimizer.apply_gradients(zip(gradient_compressed, variables))

        self.iter += 1

    def summary(self, add: str = ""):
        if self.compression is None:
            try:
                print(f"---\nOptimizer: {self.optimizer.name}\nCompression: None\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: {self.optimizer._name}\nCompression: None\n--- {add}")

        else:
            try:
                print(f"---\nOptimizer: {self.optimizer.name}\nCompression: {self.compression.name}\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: {self.optimizer._name}\nCompression: {self.compression.name}\n--- {add}")

    def get_plot_title(self):
        if self.compression is None:
            try:
                return "{} - {:.4f}".format(self.optimizer.name,
                                            self.optimizer.learning_rate.numpy())
            except AttributeError:
                return "{} - {:.4f}".format(self.optimizer._name,
                                            self.optimizer.learning_rate.numpy())

        else:
            try:
                return "{} - {} - {:.4f}".format(self.optimizer.name,
                                             self.compression.name,
                                             self.optimizer.learning_rate.numpy())
            except AttributeError:
                return "{} - {} - {:.4f}".format(self.optimizer._name,
                                                 self.compression.name,
                                                 self.optimizer.learning_rate.numpy())
    def get_file_name(self):
        if self.compression is None:
            try:
                return "{}".format(self.optimizer.name)
            except AttributeError:
                return "{}".format(self.optimizer._name)

        else:
            try:
                return "{}_{}".format(self.optimizer.name,
                                             self.compression.name)
            except AttributeError:
                return "{}_{}".format(self.optimizer._name,
                                      self.compression.name)
