import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import tensorflow_probability as tfp

from .Compression import Compression


class Atomo(Compression):
    def __init__(self, sparsity_budget: int, name="Atomo"):
        super().__init__(name=name)
        self.sparsity_budget = sparsity_budget

    def build(self, var_list):
        """Initialize optimizer variables.
        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        # if gradient.ndim != 2:
        #    gradient = self._resize_to_2d(gradient)
        # s, u, v = tf.linalg.svd(gradient, full_matrices=False)
        gradient_sparse = self.atomic_sparsification(gradient, self.sparsity_budget)
        # print(s.shape, gradient.ndim)
        # i, pi = self.atomic_sparsification(gradient, s=self.sparsity_budget)
        #
        # print(i, pi)
        return gradient_sparse

    def atomic_sparsification(self, gradient: Tensor, s: int):
        """
        Basic implementation of entry-wise decomposition: lambda_i = gradient_i
        """
        input_shape = gradient.shape
        gradient = tf.reshape(gradient, [-1]).numpy()
        # i = 0
        # n = gradient.shape[0]
        # p = tf.ones_like(gradient, dtype=gradient.dtype)
        #
        # while i <= n:
        #     if False:
        #         i = n + 1
        #     else:
        #         i += 1
        #         s -= 1
        #
        probs = gradient / gradient[0] if s == 0 else s * gradient / tf.reduce_sum(gradient)
        probs = probs.numpy()
        for i, p in enumerate(probs):
            if p > 1:
                probs[i] = 1
        # sampled_idx = []
        # sample_probs = []
        # for i, p in enumerate(probs):
        #     # if np.random.rand() < p:
        #     # random sampling from bernulli distribution
        #     if np.random.binomial(1, p):
        #         sampled_idx += [i]
        #         sample_probs += [p]
        # rank_hat = len(sampled_idx)
        # if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
        #     return self.atomic_sparsification(gradient, s=s)
        # return np.array(sampled_idx, dtype=int), np.array(sample_probs)

        t = tfp.distributions.Bernoulli(probs=probs, dtype=gradient.dtype).sample()

        gradient_estimator = gradient * t / probs
        gradient_estimator = tf.reshape(gradient_estimator, input_shape)
        return gradient_estimator

    @staticmethod
    def _resize_to_2d(x: Tensor):
        """
        x.shape > 2
        If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
        """
        shape = tf.shape(x)
        if x.ndim == 1:
            n = shape[0]
            return tf.reshape(x, (n // 2, 2))
        print([s == 1 for s in shape[2:]])
        print(shape[2:], shape.numpy())
        if tf.reduce_all([s == 1 for s in shape[2:]]):
            return tf.reshape(x, (shape[0], shape[1]))
        x = tf.reshape(x, (shape[0], shape[1], -1))
        x_tmp = tf.reshape(x, (shape[0] * shape[1], -1))
        tmp_shape = tf.shape(x_tmp)
        print(tmp_shape.numpy())
        return x_tmp  # tf.reshape(x_tmp, (tmp_shape[0] // 2, tmp_shape[1] * 2))

    def _sample_svd(self, s, rank=0):
        if s[0] < 1e-6:
            return [0], np.array([1.0])
        probs = s / s[0] if rank == 0 else rank * s / s.sum()
        for i, p in enumerate(probs):
            if p > 1:
                probs[i] = 1
        sampled_idx = []
        sample_probs = []
        for i, p in enumerate(probs):
            # if np.random.rand() < p:
            # random sampling from bernulli distribution
            if np.random.binomial(1, p):
                sampled_idx += [i]
                sample_probs += [p]
        rank_hat = len(sampled_idx)
        if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
            return self._sample_svd(s, rank=rank)
        return np.array(sampled_idx, dtype=int), np.array(sample_probs)
