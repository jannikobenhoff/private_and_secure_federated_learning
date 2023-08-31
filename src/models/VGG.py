"""
VGG11/13/16/19 in TensorFlow2.

Reference:
[1] Simonyan, Karen, and Andrew Zisserman.
    "Very deep convolutional networks for large-scale image recognition."
    arXiv preprint arXiv:1409.1556 (2014).
"""
import asyncio
from datetime import datetime

import tensorflow as tf
from keras import layers
from keras.regularizers import l2

config_codebook = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes, lambda_l2):
        super(VGG, self).__init__()
        self.conv = self._make_layers(config_codebook[vgg_name], lambda_l2=lambda_l2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x,
             training=None,
             mask=None):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    @staticmethod
    def _make_layers(config, lambda_l2):
        layer = []
        for l in config:
            if l == 'M':
                layer += [layers.MaxPool2D(pool_size=2, strides=2)]
            else:
                layer += [layers.Conv2D(l, kernel_size=3, padding='same', kernel_regularizer=l2(lambda_l2)),
                          layers.BatchNormalization(),
                          layers.ReLU()]
        layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
        return tf.keras.Sequential(layer)


if __name__ == "__main__":
    # vgg = VGG(vgg_name="vgg11", num_classes=10, lambda_l2=None)
    # vgg.build(input_shape=(None, 32, 32, 3))
    # vgg.summary()
    print(datetime.now().strftime('%m_%d_%H_%M_%S'))
