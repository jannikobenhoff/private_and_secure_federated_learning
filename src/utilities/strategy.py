import numpy as np
from keras.optimizers.legacy import optimizer_v2
from tensorflow import Tensor
import tensorflow as tf


class Strategy(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.0,
            compression=None,
            nesterov=False,
            name="SGD",
            **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        self.compression = compression
        if (
                isinstance(momentum, tf.Tensor)
                or callable(momentum)
                or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, (int, float)) and (
                momentum < 0 or momentum > 1
        ):
            raise ValueError(
                "`momentum` must be between [0, 1]. Received: "
                f"momentum={momentum} (of type {type(momentum)})."
            )
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.compression is not None:
            self.compression.build(var_list)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.compression is not None:
            grad = self.compression.compress(grad, var)

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

    def _resource_apply_sparse_duplicate_indices(
            self, grad, var, indices, **kwargs
    ):

        if self._momentum:
            return super()._resource_apply_sparse_duplicate_indices(
                grad, var, indices, **kwargs
            )
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = kwargs.get("apply_state", {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)

            return tf.raw_ops.ResourceScatterAdd(
                resource=var.handle,
                indices=indices,
                updates=-grad * coefficients["lr_t"],
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        momentum_var = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._initial_decay,
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config
# class Strategy:
#     def __init__(self, optimizer, compression=None):
#         self.optimizer = optimizer
#         self.compression = compression
#
#     def update_parameters(self, grads_and_vars: zip):
#         grads_and_vars = list(grads_and_vars)
#         gradient, variables = zip(*grads_and_vars)
#         gradient = list(gradient)
#
#         if self.compression is not None:
#             scope_name = "optimizer"
#             with tf.name_scope(scope_name):
#                 with tf.init_scope():
#                     # Lift variable creation to init scope to avoid environment
#                     # issues.
#                     self.compression.build(variables)
#         if self.compression is None:
#             self.optimizer.apply_gradients(zip(gradient, variables))
#         else:
#             gradient_compressed = []
#             for i, grad in enumerate(gradient):
#                 if False:  # huffman
#                     enc, huf, shape = self.compression.compress(grad, variables[i])
#                     dec_huf = decode_huffman(enc, huf)
#                     dec = decode_rle(dec_huf)
#                     gradient_compressed.append(tf.reshape(dec, shape))
#                 else:
#                     gradient_compressed.append(self.compression.compress(grad, variables[i]))
#
#             self.optimizer.apply_gradients(zip(gradient_compressed, variables))


    def summary(self, add: str = ""):
        if self.compression is None:
            try:
                print(f"---\nOptimizer: SGD\nCompression: None\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: SGD\nCompression: None\n--- {add}")

        else:
            try:
                print(f"---\nOptimizer: SGD\nCompression: {self.compression.name}\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: SGD\nCompression: {self.compression.name}\n--- {add}")

    def get_plot_title(self):
        if self.compression is None:
            try:
                return "SGD - {:.4f}".format(
                                            self.learning_rate.numpy())
            except AttributeError:
                return "SGD - {:.4f}".format(
                                            self.learning_rate.numpy())

        else:
            try:
                return "SGD - {} - {:.4f}".format(
                                                 self.compression.name,
                                                 self.optimizer.learning_rate.numpy())
            except AttributeError:
                return "SGD - {} - {:.4f}".format(
                                                 self.compression.name,
                                                 self.optimizer.learning_rate.numpy())

    def get_file_name(self):
        if self.compression is None:
            try:
                return "SGD"
            except AttributeError:
                return "SGD"

        else:
            try:
                return "SGD_{}".format(self.compression.name)
            except AttributeError:
                return "SGD_{}".format(
                                      self.compression.name)
