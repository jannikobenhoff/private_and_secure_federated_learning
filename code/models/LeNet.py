from keras import models, layers
import tensorflow as tf


class LeNet:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                                     activation='tanh', input_shape=input_shape, padding="same"))
        self.model.add(layers.AveragePooling2D(2))
        # self.model.add(layers.Activation('sigmoid'))
        self.model.add(layers.Conv2D(16, 5, activation='tanh'))
        self.model.add(layers.AveragePooling2D(2))
        # self.model.add(layers.Activation('sigmoid'))
        # self.model.add(layers.Conv2D(120, 5, activation='tanh'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation='tanh'))
        self.model.add(layers.Dense(84, activation='tanh'))
        self.model.add(layers.Dense(10, activation='softmax'))

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
