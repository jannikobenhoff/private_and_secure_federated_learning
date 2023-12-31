import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import tensorflow_probability as tfp

from .Compression import Compression


class Atomo(Compression):
    def __init__(self, svd_rank: int, random_sample=True, name="Atomo"):
        super().__init__(name=name)

        self.svd_rank = svd_rank
        self.random_sample = random_sample

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.
        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradients: list[Tensor], variables: list[Tensor], client_id: int = 1, log=True):
        compressed_grads = []
        for idx, gradient in enumerate(gradients):
            if gradient.ndim != 2:
                gradient_ndim2 = self._resize_to_2d(gradient)
            else:
                gradient_ndim2 = gradient

            s, u, vT = tf.linalg.svd(gradient_ndim2, full_matrices=False)

            vT = tf.transpose(vT).numpy()
            s = s.numpy()
            u = u.numpy()

            if self.random_sample:
                i, probs = self._sample_svd(s, rank=self.svd_rank)
                u = u[:, i]
                s = s[i] / probs
                vT = vT[i, :]
            elif self.svd_rank > 0:
                u = u[:, :self.svd_rank]
                s = s[:self.svd_rank]
                vT = vT[:self.svd_rank, :]

            if log:  # variables[idx].ref() not in self.cr:
                grad_type = gradient.dtype.size
                bit_size = (np.prod(
                    u.shape) + np.prod(
                    s.shape) + np.prod(
                    vT.shape)) * grad_type * 8

                # self.cr[variables[idx].ref()] = grad_type * 8 * np.prod(
                #     gradient.shape.as_list()) / bit_size

                self.compression_rates.append(grad_type * 8 * np.prod(
                    gradient.shape.as_list()) / bit_size)
                self.compression_rates = [np.mean(self.compression_rates)]

            compressed_grads.append({
                "u": u,
                "s": s,
                "vT": vT
            })
        # eventuell orig size hinzufügen bei decompression_info
        return {
            "compressed_grads": compressed_grads,
            "decompression_info": None,
            "needs_decompress": True
        }

    def decompress(self, compressed_data, variables):
        decompressed_grads = []
        for i, _ in enumerate(compressed_data["compressed_grads"]):
            s = compressed_data["compressed_grads"][i]["s"]
            u = compressed_data["compressed_grads"][i]["u"]
            vT = compressed_data["compressed_grads"][i]["vT"]
            decoded = tf.convert_to_tensor(np.dot(np.dot(u, np.diag(s)), vT))
            decoded = tf.reshape(decoded, variables[i].shape)
            decoded = tf.cast(decoded, dtype=variables[i].dtype)
            decompressed_grads.append(decoded)

        return decompressed_grads

    def _sample_svd(self, s, rank=0):
        if s[0] < 1e-6:
            return [0], np.array([1.0])
        probs = s / s[0] if rank == 0 else rank * s / s.sum()
        for i, p in enumerate(probs):
            if p > 1:
                probs[i] = 1
        # print("Probs:", np.sum(probs))  # =!= rank
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
        if tf.reduce_all([s == 1 for s in shape[2:]]):
            return tf.reshape(x, (shape[0], shape[1]))
        x = tf.reshape(x, (shape[0], shape[1], -1))
        x_tmp = tf.reshape(x, (shape[0] * shape[1], -1))
        tmp_shape = tf.shape(x_tmp)
        # print(tmp_shape.numpy())
        return x_tmp
        # return tf.reshape(x_tmp, (tmp_shape[0] // 2, tmp_shape[1] * 2))
