import tensorflow as tf
from keras.layers import Dense, Flatten, AveragePooling2D, Conv2D
from tensorflow import keras
from keras import layers, models, regularizers, Model, Input


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


def LeNet5(input_shape=(int, int, int), l2_lambda=None):
    inputs = Input(shape=input_shape)

    # convolutional layer 1
    model_layers = Conv2D(
        kernel_size=5,
        filters=6,
        padding='same',
        activation=tf.nn.tanh,
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(inputs)
    # pooling layer 1
    model_layers = AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(model_layers)
    # convolutional layer 2
    model_layers = Conv2D(
        kernel_size=5,
        filters=16,
        padding='valid',
        activation=tf.nn.tanh,
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(model_layers)
    # pooling layer 2
    model_layers = AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(model_layers)

    model_layers = Flatten()(model_layers)
    # fully connected layers(120, 84, 10)
    model_layers = Dense(
        units=120,
        activation=tf.nn.tanh,
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(model_layers)
    model_layers = Dense(
        units=84,
        activation=tf.nn.tanh,
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(model_layers)
    outputs = Dense(
        units=10,
        activation='softmax',
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(model_layers)

    model = Model(inputs, outputs)

    return model
