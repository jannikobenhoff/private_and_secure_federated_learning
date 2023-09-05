import tensorflow as tf
from keras.optimizers.optimizer import Optimizer
from tensorflow import Tensor


class EFsignSGD(Optimizer):
    def __init__(self, learning_rate, name="EFsignSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.compression_rates = []

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        EFsignSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build EFsignSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        if clients == 1:
            self.errors = []
            for var in var_list:
                self.errors.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="error"
                    )
                )
        else:
            self.errors = {}
            # Federated Setup
            for client_id in range(1, clients + 1):
                for var in var_list:
                    self.errors[var.name + str(client_id)] = self.add_variable_from_reference(
                        model_variable=var, variable_name="error", initial_value=tf.zeros_like(var)
                    )
        self.compression_rates.append(var_list[0].dtype.size * 8)
        self._built = True

    def update_step(self, gradient: Tensor, variable, lr) -> Tensor:
        """
        Send sign(p_t) and norm(p_t)
        -> compression rate like signSGD + bits for norm
        """
        self.lr = lr
        lr = tf.cast(self.lr, variable.dtype.base_dtype)

        var_key = self._var_key(variable)

        error = self.errors[self._index_dict[var_key]]

        d = tf.size(gradient)
        d = tf.cast(d, dtype=gradient.dtype)

        p_t = lr * gradient + error
        norm = tf.divide(tf.norm(p_t, ord=1), d)
        delta_t = tf.multiply(norm, tf.sign(p_t))

        # update residual error
        self.errors[self._index_dict[var_key]].assign(p_t - delta_t)

        return delta_t

    def federated_compress(self, gradients, variables, client_id, lr):
        compressed_gradients = []
        norm_d = []
        for i, gradient in enumerate(gradients):
            error = self.errors[variables[i].name + str(client_id)]

            d = tf.size(gradient)
            d = tf.cast(d, dtype=gradient.dtype)

            p_t = lr * gradient + error
            norm = tf.divide(tf.norm(p_t, ord=1), d)
            delta_t = tf.multiply(norm, tf.sign(p_t))

            # update residual error
            self.errors[variables[i].name + str(client_id)].assign(p_t - delta_t)

            compressed_gradients.append(tf.sign(p_t))
            norm_d.append(norm)

        return {
            "compressed_grad": compressed_gradients,
            "decompress_info": norm_d
        }

    def federated_decompress(self, info, variables, lr):
        decompressed_gradients = []
        for i, gradient in enumerate(info["compressed_grad"]):
            norm_d = info["decompress_info"][i]
            decompressed_gradients.append(tf.multiply(gradient, norm_d) / lr)

        return decompressed_gradients

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                )
            }
        )
        return config
