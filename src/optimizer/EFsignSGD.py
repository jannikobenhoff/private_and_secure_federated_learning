import tensorflow as tf
from keras.optimizers.optimizer import Optimizer
from tensorflow import Tensor


class EFsignSGD(Optimizer):
    def __init__(self, learning_rate, name="EFsignSGD"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.compression_rates = []
        self.errors = {}

    def build(self, var_list, clients=1):
        """Initialize optimizer variables.

        EFsignSGD optimizer has one variable `quantization error`

        Args:
          var_list: list of model variables to build EFsignSGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        for client in range(1, clients + 1):
            self.errors[str(client)] = tf.Variable(tf.zeros(shape=tf.reduce_sum([tf.size(var) for var in var_list])))
        self.compression_rates.append(var_list[0].dtype.size * 8)
        self._built = True

    def compress(self, grads: list[Tensor], variables: list[Tensor], lr, client_id: int = 1):
        flattened_grads = [tf.reshape(grad, [-1]) for grad in grads]
        gradient = tf.concat(flattened_grads, axis=0)

        error = self.errors[str(client_id)]

        d = tf.size(gradient)
        d = tf.cast(d, dtype=gradient.dtype)

        p_t = lr * gradient + error
        norm = tf.divide(tf.norm(p_t, ord=1), d)
        delta_t = tf.multiply(norm, tf.sign(p_t))

        # update residual error
        self.errors[str(client_id)].assign(p_t - delta_t)

        # Norm is divided by lr because lr will be multiplied later in keras process
        return {
            "compressed_grads": tf.sign(p_t),
            "decompress_info": norm / lr,
            "needs_decompress": True
        }

    def decompress(self, compressed_data, variables):
        scaled_gradient = tf.multiply(compressed_data["compressed_grads"], compressed_data["decompress_info"])

        decompressed_grads = []
        start = 0
        for var in variables:
            size = tf.reduce_prod(var.shape).numpy()
            segment = tf.Variable(scaled_gradient[start: start + size])
            decompressed_grads.append(tf.reshape(segment, var.shape))
            start += size
        return decompressed_grads

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
