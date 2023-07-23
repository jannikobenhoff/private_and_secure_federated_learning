import tensorflow as tf
from tensorflow import Tensor

from src.compressions.Compression import Compression


class vqSGD(Compression):
    def __init__(self, momentum, name="vqSGD"):
        super().__init__(name=name)
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables.

        vqSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build vqSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self.error = {}
        for var in var_list:
            self.error[var.ref()] = self.add_variable_from_reference(
                model_variable=var, variable_name="error",
                initial_value=tf.zeros(shape=tf.shape(var))
            )
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:

        # update residual error
        self.error[variable.ref()].assign()

        # update iterate
        variable.assign_add()
        return gradient
