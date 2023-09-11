import tensorflow as tf
from keras.regularizers import l2
from keras import layers

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from tensorflow import Tensor
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model


def resnet(num_filters: int, size: int, input_shape=(int, int, int), lambda_l2=None):
    inputs = Input(shape=input_shape)

    # First Layer
    model_layers = BatchNormalization()(inputs)
    model_layers = Conv2D(kernel_size=3,
                          strides=1,
                          filters=num_filters,
                          padding="same",
                          kernel_regularizer=l2(lambda_l2))(model_layers)

    model_layers = ReLU()(model_layers)
    model_layers = BatchNormalization()(model_layers)

    # First size/8, second size/4, third size/8, last size/8
    block_length_array = [math.floor(size / 8), math.ceil(size / 4), math.ceil(size / 4), math.floor(size / 8)]

    for layer_block_num in range(len(block_length_array)):

        if layer_block_num == 0:
            for k in range(block_length_array[layer_block_num]):
                model_layers = residual_block(model_layers, downsample=False, filters=num_filters, lambda_l2=lambda_l2)

        else:
            for k in range(block_length_array[layer_block_num]):
                model_layers = residual_block(model_layers, downsample=(k == 0), filters=num_filters,
                                              lambda_l2=lambda_l2)
        num_filters *= 2

    model_layers = AveragePooling2D(4)(model_layers)
    model_layers = Flatten()(model_layers)
    outputs = Dense(10, activation='softmax')(model_layers)
    model = Model(inputs, outputs)
    # model.compile(
    #     optimizer=optimizer,
    #     loss=loss,
    #     metrics=metric)
    return model


def residual_block(input: Tensor, downsample: bool, filters: int, lambda_l2):
    block_layer = Conv2D(kernel_size=3,
                         strides=(2 if downsample else 1),
                         filters=filters,
                         padding="same",
                         kernel_regularizer=l2(lambda_l2))(input)

    block_layer = ReLU()(block_layer)
    block_layer = BatchNormalization()(block_layer)

    block_layer = Conv2D(kernel_size=3,
                         strides=1,
                         filters=filters,
                         padding="same",
                         kernel_regularizer=l2(lambda_l2))(block_layer)

    if downsample:
        input = Conv2D(kernel_size=1,
                       strides=2,
                       filters=filters,
                       padding="same",
                       kernel_regularizer=l2(lambda_l2))(input)

    block_layer = Add()([input, block_layer])

    relu = ReLU()(block_layer)
    block_layer = BatchNormalization()(relu)

    return block_layer


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, lambda_l2, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False,
                                   kernel_regularizer=l2(lambda_l2))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_regularizer=l2(lambda_l2)
                                   )
        self.bn2 = layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion * out_channels, kernel_size=1, strides=strides, use_bias=False,
                              kernel_regularizer=l2(lambda_l2)
                              ),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x

    def call(self, x, training=None, mask=None):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out


class BottleNeck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, lambda_l2, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False,
                                   kernel_regularizer=l2(lambda_l2))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False,
                                   kernel_regularizer=l2(lambda_l2))
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * out_channels, kernel_size=1, use_bias=False,
                                   kernel_regularizer=l2(lambda_l2))
        self.bn3 = layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion * out_channels, kernel_size=1, strides=strides, use_bias=False,
                              kernel_regularizer=l2(lambda_l2)),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x

    def call(self, x, training=None, mask=None):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out


class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, lambda_l2):
        super(BuildResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                                   kernel_regularizer=l2(lambda_l2))
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], lambda_l2, strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], lambda_l2, strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], lambda_l2, strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], lambda_l2, strides=2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=None, mask=None):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, lambda_l2, strides):
        stride = [strides] + [1] * (num_blocks - 1)
        layer = []
        for s in stride:
            layer += [block(self.in_channels, out_channels, lambda_l2, s)]
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layer)


def ResNet(model_type, num_classes, lambda_l2):
    """
    Implementation by https://github.com/lionelmessi6410/tensorflow2-cifar

    Reference:
    [1] He, Kaiming, et al.
    "Deep residual learning for image recognition."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

    """
    if model_type == 'resnet18':
        return BuildResNet(BasicBlock, [2, 2, 2, 2], num_classes, lambda_l2)
    elif model_type == 'resnet34':
        return BuildResNet(BasicBlock, [3, 4, 6, 3], num_classes, lambda_l2)
    elif model_type == 'resnet50':
        return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes, lambda_l2)
    elif model_type == 'resnet101':
        return BuildResNet(BottleNeck, [3, 4, 23, 3], num_classes, lambda_l2)
    elif model_type == 'resnet152':
        return BuildResNet(BottleNeck, [3, 8, 36, 3], num_classes, lambda_l2)
    else:
        ValueError("{:s} is currently not supported.".format(model_type))


if __name__ == "__main__":
    a = ResNet("resnet18", 10, None)
    a.build(input_shape=(None, 32, 32, 3))
    a.summary()
