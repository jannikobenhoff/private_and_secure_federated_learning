import tensorflow as tf
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow import Tensor
from src.compressions.TopK import TopK


class SGD(Optimizer):
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

    def _resource_apply_dense(self, grad, var, apply_state=None):
        grad = TopK(k=10).compress(grad, var)
        print(grad[0:10])
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return tf.raw_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=momentum_var.handle,
                lr=coefficients["lr_t"],
                grad=grad,
                momentum=coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov,
            )
        else:
            return tf.raw_ops.ResourceApplyGradientDescent(
                var=var.handle,
                alpha=coefficients["lr_t"],
                delta=grad,
                use_locking=self._use_locking,
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        gradient = TopK(k=10).compress(gradient, variable)
        print(gradient[0:10])
        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]
        if gradient.dtype != variable.dtype:
            gradient = tf.cast(gradient, dtype=variable.dtype)
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
