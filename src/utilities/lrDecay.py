import tensorflow as tf


class CosineDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, decay_steps, data_length, batch_size, alpha):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.data_length = data_length
        self.batch_size = batch_size
        self.alpha = alpha
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            alpha=alpha
        )

    def on_epoch_begin(self, epoch, logs=None):
        """
        decay learning rate = alpha + (1-alpha) * (1 + cos(pi * global_step/decay_steps)/2
        """
        new_lr = self.cosine_decay(epoch*self.data_length/self.batch_size)
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print('CosineDecay setting learning rate to {:.4f}'.format(new_lr.numpy()))
