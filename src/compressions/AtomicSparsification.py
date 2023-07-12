import numpy as np
import tensorflow as tf

class AtomicSparsification:
    def __init__(self, dictionary, lr):
        self.dictionary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # self.dictionary = dictionary
        self.lr = lr

    def update_step(self, gradient, variable):
        lr = tf.cast(self.lr, variable.dtype)

    def sparsify(self, signal, sparsity):
        inner_products = np.dot(signal, self.dictionary.T)

        # Select the top-k atoms with the largest inner products
        top_k_indices = np.argsort(inner_products)[-sparsity:]

        # Create the sparse representation using the selected atoms
        sparse_representation = np.zeros_like(signal)
        sparse_representation[top_k_indices] = inner_products[top_k_indices]

        return sparse_representation

