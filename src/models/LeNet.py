from keras import models, layers, regularizers
import tensorflow as tf


class LeNet:
    def build_model(self, hp):
        input_shape = (28, 28, 1)
        lambda_start = 0.001
        lambda_end = 0.01
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                                     activation='tanh', input_shape=input_shape, padding="same",
                                     kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2))
        # self.model.add(layers.Activation('sigmoid'))
        self.model.add(layers.Conv2D(16, 5, activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2))
        # self.model.add(layers.Activation('sigmoid'))
        # self.model.add(layers.Conv2D(120, 5, activation='tanh'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(84, activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(0.001),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        return self.model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def fit(self, ds_train, epochs, validation_data, callbacks):
        self.model.fit(ds_train,
                       epochs=epochs,
                       validation_data=validation_data,
                       callbacks=callbacks)

    def summary(self):
        self.model.summary()
