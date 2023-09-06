import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers


def LeNet(input_shape, num_classes, l2_lambda):
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
