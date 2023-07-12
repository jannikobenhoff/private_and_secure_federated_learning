from keras import models, layers, regularizers
import tensorflow as tf


class LeNet:
    def __init__(self):
        self.model = None

    def build_model(self, hp):
        input_shape = (28, 28, 1)
        lambda_start = 1e-6
        lambda_end = 0.001
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, kernel_size=(5, 5),
                                     activation='tanh', input_shape=input_shape,
                                     kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        #self.model.add(layers.Activation('sigmoid'))
        self.model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.AveragePooling2D(2, strides=2))
        #self.model.add(layers.Activation('sigmoid'))
        #self.model.add(layers.Conv2D(120, kernel_size=(5, 5), activation='tanh',  kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(84, activation='tanh', kernel_regularizer=regularizers.l2(hp.Float('lambda', lambda_start, lambda_end, sampling='log'))))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    # def compile(self, optimizer, loss, metrics):
    #     self.model.compile(optimizer=optimizer,
    #                        loss=loss,
    #                        metrics=metrics)
    #
    # def fit(self, ds_train, epochs, validation_data, callbacks):
    #     self.model.fit(ds_train,
    #                    epochs=epochs,
    #                    validation_data=validation_data,
    #                    callbacks=callbacks)

    def summary(self):
        self.model.summary()
