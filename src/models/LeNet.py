import tensorflow as tf
from keras import layers, models, regularizers


class LeNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes, chosen_lambda, search: bool = False):
        super(LeNet, self).__init__()
        self.model = None
        self.shape = input_shape
        self.num_classes = num_classes
        self.chosen_lambda = chosen_lambda
        if not search:
            self._build_model()

    def _build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=self.shape,
                                     kernel_regularizer=regularizers.l2(self.chosen_lambda)))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        self.model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh',
                                     kernel_regularizer=regularizers.l2(self.chosen_lambda)))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(self.chosen_lambda)))
        self.model.add(layers.Dense(84, activation='tanh', kernel_regularizer=regularizers.l2(self.chosen_lambda)))
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))

    def search_model(self, hp):
        input_shape = (28, 28, 1)
        lambda_start = 1e-7
        lambda_end = 0.1
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, kernel_size=(5, 5),
                                     activation='tanh', input_shape=input_shape,
                                     kernel_regularizer=regularizers.l2(
                                         hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        # self.model.add(layers.Activation('sigmoid'))
        self.model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh', kernel_regularizer=regularizers.l2(
            hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        # self.model.add(layers.Activation('sigmoid'))
        # self.model.add(layers.Conv2D(120, kernel_size=(5, 5), activation='tanh',  kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(
            hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(84, activation='tanh', kernel_regularizer=regularizers.l2(
            hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(10, activation='softmax'))

        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(0.001),
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy']
        # )

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
