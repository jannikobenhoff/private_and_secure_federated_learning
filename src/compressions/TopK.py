import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from .Compression import Compression


class TopK(Compression):
    def __init__(self, k, name="TopK"):
        super().__init__(name=name)
        self.k = k
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        TernGrad optimizer has no additional variables.

        Args:
          var_list: list of model variables to build OneBitSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        sparse_gradient = self.top_k_sparsification(gradient, self.k)

        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                sparse_gradient)
            self.compression_rates.append(self.cr[variable.ref()])
            print(np.mean(self.compression_rates))

        return sparse_gradient
