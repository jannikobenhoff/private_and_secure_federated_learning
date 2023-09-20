import numpy as np
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow import Tensor
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


class Strategy(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.0,
            optimizer="sgd",
            params=None,
            compression=None,
            nesterov=False,
            name="Strategy",
            **kwargs,
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        self.optimizer_name = optimizer
        self.compression = compression
        self.compression_name = "None"
        if compression is not None:
            self.compression_name = compression.name
        self.learning_rate = learning_rate
        self.num_clients = 1

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
        self.params = params
        # self.optimizer = None
        #
        # if self.optimizer_name == "efsignsgd":
        #     self.compression = EFsignSGD(learning_rate=learning_rate)
        # if self.optimizer_name == "fetchsgd":
        #     self.compression = FetchSGD(learning_rate=learning_rate, c=params["c"], r=params["r"],
        #                                 momentum=params["momentum"], topk=params["topk"])
        # if self.optimizer_name == "memsgd":
        #     if params["top_k"] == "None":
        #         self.compression = MemSGD(learning_rate=learning_rate, rand_k=params["rand_k"])
        #     else:
        #         self.compression = MemSGD(learning_rate=learning_rate, top_k=params["top_k"])
        #
        # self.optimizer = SGD(learning_rate=learning_rate)

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.compression is not None:
            self.compression.build(var_list)
        # if self.optimizer_name != "sgd":
        #     self.compression.build(var_list)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def build(self, variables, num_clients=1):
        self.num_clients = num_clients
        if self.compression is not None:
            self.compression.build(variables, num_clients)
        # elif self.optimizer_name != "sgd":
        #     self.compression.build(variables, num_clients)

    def compress(self, grads, variables, client_id=1, number_clients=1):
        """
        Gradients from loss function and trainable model weights
        """
        # if self.optimizer_name != "sgd":
        #     compressed_data = self.compression.compress(grads, variables, client_id=client_id, lr=self.learning_rate)
        #     return compressed_data
        if self.compression is not None:
            compressed_data = self.compression.compress(grads, variables, client_id=client_id, lr=self.learning_rate)
            return compressed_data
        else:
            return {
                "compressed_grads": grads,
                "decompress_info": None,
                "needs_decompress": False
            }

    def decompress(self, compressed_data: dict, variables):
        if "client_1" in compressed_data.keys():
            # Federated Learning, every entry is compressed data by client.
            if not compressed_data["client_1"]["needs_decompress"]:
                for client in range(1, len(compressed_data) + 1):
                    client = "client_" + str(client)

                    client_grads = tf.cond(tf.equal(client, "client_1"),
                                           lambda: compressed_data[client]["compressed_grads"],
                                           lambda: tf.nest.map_structure(lambda x, y: x + y, client_grads,
                                                                         compressed_data[client]["compressed_grads"]))
                # Average Gradients
                client_grads = tf.nest.map_structure(lambda x: x / len(compressed_data.keys()), client_grads)
                return client_grads
            elif self.compression_name == "fetchsgd":
                for client in range(1, len(compressed_data) + 1):
                    client = "client_" + str(client)
                    client_sketches = tf.cond(tf.equal(client, "client_1"),
                                              lambda: compressed_data[client]["compressed_grads"],
                                              lambda: tf.nest.map_structure(lambda x, y: x + y, client_sketches,
                                                                            compressed_data[client][
                                                                                "compressed_grads"]))
                # Average Sketches
                client_sketches = tf.nest.map_structure(lambda x: x / len(compressed_data.keys()), client_sketches)
                decompressed_grads = self.compression.decompress({"compressed_grads": client_sketches}, variables)
                return decompressed_grads
            # elif self.optimizer_name != "sgd":
            #     for client in range(1, len(compressed_data) + 1):
            #         client = "client_" + str(client)
            #         client_grads = tf.cond(tf.equal(client, "client_1"),
            #                                lambda: self.compression.decompress(compressed_data[client], variables),
            #                                lambda: tf.nest.map_structure(lambda x, y: x + y, client_grads,
            #                                                              self.compression.decompress(
            #                                                                  compressed_data[client], variables)))
            #     # Average Gradients
            #     client_grads = tf.nest.map_structure(lambda x: x / len(compressed_data.keys()), client_grads)
            #     return client_grads
            elif self.compression is not None:
                for client in range(1, len(compressed_data) + 1):
                    client = "client_" + str(client)
                    client_grads = tf.cond(tf.equal(client, "client_1"),
                                           lambda: self.compression.decompress(compressed_data[client], variables),
                                           lambda: tf.nest.map_structure(lambda x, y: x + y, client_grads,
                                                                         self.compression.decompress(
                                                                             compressed_data[client], variables)))
                # Average Gradients
                client_grads = tf.nest.map_structure(lambda x: x / len(compressed_data.keys()), client_grads)
                return client_grads
        else:
            if self.compression is not None:
                decompressed_grads = self.compression.decompress(compressed_data, variables)
                return decompressed_grads
            # elif self.optimizer_name != "sgd":
            #     decompressed_grads = self.compression.decompress(compressed_data, variables)
            #     return decompressed_grads

    def _resource_apply_dense(self, grad, var, apply_state=None):
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

    def compression_rates(self):
        if self.compression is not None:
            return self.compression.compression_rates
        # elif self.optimizer_name != "sgd":
        #     return self.optimizer.compression_rates
        else:
            return 1

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

    def summary(self, add: str = ""):
        if self.compression is None:
            try:
                print(f"---\nOptimizer: {self.optimizer_name.upper()}\nCompression: None\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: {self.optimizer_name.upper()}\nCompression: None\n--- {add}")

        else:
            try:
                print(f"---\nOptimizer: {self.optimizer_name.upper()}\nCompression: {self.compression.name}\n--- {add}")
            except AttributeError:
                print(f"---\nOptimizer: {self.optimizer_name.upper()}\nCompression: {self.compression.name}\n--- {add}")

    def get_plot_title(self):
        if self.compression is None:
            return "{} - {:.4f}".format(self.optimizer_name.upper(),
                                        self.learning_rate.numpy())

        else:
            return "{} - {} - {:.4f}".format(self.optimizer_name.upper(),
                                             self.compression.name,
                                             self.learning_rate.numpy())

    def get_file_name(self):
        if self.compression is None:
            add_on = ""
            for key in self.params.keys():
                if key != 'optimizer' and key != 'compression' and key != 'learning_rate':
                    add_on += "_" + key + str(self.params[key])
            return "{}{}".format(self.optimizer_name.upper(), add_on)
        else:
            add_on = ""
            for key in self.params.keys():
                if key != 'optimizer' and key != 'compression' and key != 'learning_rate':
                    add_on += "_" + key + str(self.params[key])
            return "{}_{}{}".format(self.optimizer_name.upper(), self.compression.name, add_on)
