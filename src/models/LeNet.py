import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers


class LeNet(tf.keras.Model):
    def __init__(self, input_shape = None, num_classes=None, chosen_lambda= None, search: bool = False):
        super(LeNet, self).__init__()
        #self.model = None
        self.shape = input_shape
        self.num_classes = num_classes
        self.chosen_lambda = chosen_lambda
        if not search:
            self._build_model()
            print("LeNet built.")

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

    @staticmethod
    def search_model(lambda_l2):
        input_shape = (28, 28, 1)

        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                                kernel_regularizer=regularizers.l2(lambda_l2)))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                                kernel_regularizer=regularizers.l2(lambda_l2)))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)))
        model.add(layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)))
        model.add(layers.Dense(10, activation='softmax'))

        # model = tf.keras.models.Sequential()
        # model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        # model.add(layers.MaxPooling2D())
        # model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D())
        # model.add(layers.Flatten())
        # model.add(layers.Dense(120, activation='relu'))
        # model.add(layers.Dense(84, activation='relu'))
        # model.add(layers.Dense(10, activation='softmax'))
        return model

    # def call(self, inputs, training=None, mask=None):
    #     return self.model(inputs)
