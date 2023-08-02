import tensorflow as tf
from tensorflow import Tensor
from src.compressions import Compression


class ScaleCom(Compression):
    def __init__(self, discount: float, k: int, name="scaleCom"):
        super().__init__(name=name)
        self.compression_rates = []
        self.discount = discount
        self.k = k

    def build(self, var_list):
        """Initialize optimizer variables.

        vqSGD optimizer has no variables.

        Args:
          var_list: list of model variables to build vqSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.memory = {}
        for var in var_list:
            self.memory[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="memory", initial_value=tf.zeros_like(var)
            )
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        m = self.memory[variable.ref()]
        discount = tf.cast(self.discount, variable.dtype.base_dtype)

        compressed_gradient = self.cltk_compression(input_tensor=m+gradient, k=self.k)

        self.memory[variable.ref()].assign((1-discount)*m+discount*(m+gradient-compressed_gradient))

        return compressed_gradient

    @staticmethod
    def cltk_compression(input_tensor: Tensor, k: int):
        """
        Cyclic Local Top-k Compressor, same as top-k if only one worker.
        Leading worker sends its top-k indices to other workers.
        """
        input_shape = input_tensor.shape
        flattened_tensor: Tensor = tf.reshape(input_tensor, [-1])

        if tf.size(flattened_tensor) < k:
            return input_tensor

        abs_tensor = tf.abs(flattened_tensor)
        indices = tf.math.top_k(abs_tensor, k).indices

        mask = tf.zeros(flattened_tensor.shape)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=1),
                                           tf.ones_like(indices, dtype=tf.float32))

        spars_tensor = flattened_tensor * mask
        spars_tensor = tf.reshape(spars_tensor, input_shape)

        return spars_tensor