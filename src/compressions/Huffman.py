import tensorflow as tf
from tensorflow import Tensor

from src.compressions.Compression import Compression


class Huffman(Compression):
    def __init__(self, name="Huffman"):
        super().__init__(name=name)

    def build(self, var_list):
        """Initialize optimizer variables.
        Huffman optimizer has no variables.

        Args:
          var_list: list of model variables to build Huffman variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        return gradient
