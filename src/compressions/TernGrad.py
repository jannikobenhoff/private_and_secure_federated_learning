import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer
from tensorflow import Tensor


class TernGrad(optimizer.Optimizer):
    def __init__(self, learning_rate, c: float, name="TernGrad"):
        super().__init__(name=name)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.c = c

    def build(self, var_list):
        """Initialize optimizer variables.

        TernGrad optimizer has no variable.

        Args:
          var_list: list of model variables to build TernGrad variables on.
        """
        return

    def _update_step(self, gradient: Tensor, variable):
        lr = tf.cast(self.lr, variable.dtype.base_dtype)
        gradient_clip = self.gradient_clipping(gradient, self.c)
        gradient_tern = self.ternarize(gradient_clip)

        variable.assign_add(-gradient_tern * lr)

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

    @staticmethod
    def ternarize(input_tensor: Tensor) -> Tensor:
        """
        Layer-wise ternarize

        g_t_i_tern = s_t * sign(g_t_i) o b_t
        s_t = max(abs(g_t_i)) = ||g_t_i||∞ (max norm)
        o : Hadamard product
        """
        if len(input_tensor.shape) == 1:
            abs_input = tf.abs(input_tensor)
            s_t = tf.reduce_max(abs_input, axis=0, keepdims=True)
            b_t = tf.cast(abs_input / s_t >= 0.5, input_tensor.dtype)
            return s_t * tf.sign(input_tensor) * b_t
        abs_input = tf.abs(input_tensor)
        s_t = tf.reduce_max(abs_input, axis=1, keepdims=True)
        b_t = tf.cast(abs_input / s_t >= 0.5, input_tensor.dtype)
        return s_t * tf.sign(input_tensor) * b_t

        # output_list = []
        # for layer in range(input_tensor.shape[0]):
        #     g_t_i = input_tensor[layer]
        #     s_t = tf.math.reduce_max(tf.abs(g_t_i))
        #     b_t = tf.where(tf.abs(g_t_i) / s_t >= 0.5, tf.ones_like(g_t_i), tf.zeros_like(g_t_i))
        #     output_list.append(s_t * tf.sign(g_t_i) * b_t)
        # input_shape = input_tensor.shape
        # g_t_i = tf.reshape(input_tensor, [-1])
        # s_t = tf.reduce_max(tf.abs(g_t_i), 0)
        # b_t = tf.where(tf.abs(g_t_i) / s_t >= 0.5, tf.ones_like(g_t_i), tf.zeros_like(g_t_i))
        # g_t_i_tern = s_t * tf.sign(g_t_i) * b_t
        # return tf.stack(output_list) #tf.reshape(g_t_i_tern, input_shape)

    @staticmethod
    def gradient_clipping(input_tensor: Tensor, c) -> Tensor:
        """
        Clips the gradient tensor.
        Sigma is the standard deviation of the gradient vector
        c is a constant
        """
        sigma = tf.math.reduce_std(input_tensor)
        abs_input = tf.abs(input_tensor)
        clipped_gradient = tf.where(
            abs_input <= c * sigma, input_tensor, tf.sign(input_tensor) * c * sigma)
        return clipped_gradient

    def step(self, z_t_i):
        """
        Input: z_t_i, a part of a mini-batch of training samples z_t

        Compute gradients g_t_i under z_t_i
        Ternarize gradients to g_t_i_tern = ternarize(g_t_i)
        Push ternary g_t_i_tern to the server
        Pull averaged gradients g_t_average from the server
        Update parameters w_t_next = w_t - step * g_t_average
        """
        pass


class CustomSGD(optimizer.Optimizer):

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGD",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]
        # print("update")
        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(
                -gradient.values * lr, gradient.indices
            )
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Dense gradients
            if m is not None:
                m.assign(-gradient * lr + m * momentum)
                if self.nesterov:
                    variable.assign_add(-gradient * lr + m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-gradient * lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config
