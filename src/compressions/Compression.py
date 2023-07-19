import abc
import tensorflow as tf
from tensorflow import Tensor
class Compression:
    def __init__(self, name: str):
        self.name = name
        self._variables = []

    @abc.abstractmethod
    def build(self, var_list):
        """Initialize the optimizer's variables, such as momemtum variables.

        This function has to be implemented by subclass optimizers, and subclass
        optimizers need to call `super().build(var_list)`.

        Args:
          var_list: List of model variables to build optimizers on. For example,
            SGD optimizer with momentum will store one momentum variable
            corresponding to each model variable.
        """
        if getattr(self, "_built", False):
            return
        self._build_index_dict(var_list)

    def add_variable_from_reference(
        self, model_variable, variable_name, shape=None, initial_value=None
    ):
        """Create an optimizer variable from model variable.

        Create an optimizer variable based on the information of model variable.
        For example, in SGD optimizer momemtum, for each model variable, a
        corresponding momemtum variable is created of the same shape and dtype.

        Args:
          model_variable: tf.Variable. The corresponding model variable to the
            optimizer variable to be created.
          variable_name: String. The name prefix of the optimizer variable to be
            created. The create variables name will follow the pattern
            `{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
          shape: List or Tuple, defaults to None. The shape of the optimizer
            variable to be created. If None, the created variable will have the
            same shape as `model_variable`.
          initial_value: A Tensor, or Python object convertible to a Tensor,
            defaults to None. The initial value of the optimizer variable, if
            None, the initial value will be default to 0.

        Returns:
          An optimizer variable.
        """
        if initial_value is None:
            if shape is None:
                if model_variable.shape.rank is None:
                    # When the rank is None, we cannot get a concrete
                    # `model_variable.shape`, we use dynamic shape.
                    initial_value = tf.zeros_like(
                        model_variable, dtype=model_variable.dtype
                    )
                else:
                    # We cannot always use `zeros_like`, because some cases
                    # the shape exists while values don't.
                    initial_value = tf.zeros(
                        model_variable.shape, dtype=model_variable.dtype
                    )
            else:
                initial_value = tf.zeros(shape, dtype=model_variable.dtype)
        variable = tf.Variable(
            initial_value=initial_value,
            name=f"{variable_name}/{model_variable._shared_name}",
            dtype=model_variable.dtype,
            trainable=False,
        )
        self._variables.append(variable)
        return variable

    def compress(self, gradient: Tensor, variable):
        raise NotImplementedError("Subclasses must implement compress method")

    def decompress(self, compressed_gradient: Tensor):
        raise NotImplementedError("Subclasses must implement decompress method")