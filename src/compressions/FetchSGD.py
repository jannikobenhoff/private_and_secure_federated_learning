import numpy as np
import tensorflow as tf


class FetchSGD:
    def __init__(self, lr):
        self.lr = lr

    def update_step(self, gradient, variable):
        lr = tf.cast(self.lr, variable.dtype)

        variable.assign_add(-gradient * lr)
