"""
MobileNet in TensorFlow2 by https://github.com/lionelmessi6410/tensorflow2-cifar/blob/main/models/mobilenet.py

Reference:
[1] Howard, Andrew G., et al.
    "Mobilenets: Efficient convolutional neural networks for mobile vision applications."
    arXiv preprint arXiv:1704.04861 (2017).
"""
import tensorflow as tf
from keras import layers
from keras.regularizers import l2


class Block(tf.keras.Model):
    """
    Depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, out_channels, lambda_l2, strides=1):
        super(Block, self).__init__()
        self.conv1 = layers.Conv2D(in_channels, kernel_size=3, strides=strides, padding='same',
                                   groups=in_channels, use_bias=False, kernel_regularizer=l2(lambda_l2))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False, kernel_regularizer=l2(lambda_l2))
        self.bn2 = layers.BatchNormalization()

    def call(self, x, t=None, a=None):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = tf.keras.activations.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(tf.keras.Model):
    # (128, 2) represents convolution layer with filters=128, strides=2
    config = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes, lambda_l2):
        super(MobileNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2(lambda_l2))
        self.bn1 = layers.BatchNormalization()
        self.layer = self._make_layers(in_channels=32, lambda_l2=lambda_l2)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x, t=None, a=None):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def _make_layers(self, in_channels, lambda_l2):
        layer = []
        for c in self.config:
            out_channels = c if isinstance(c, int) else c[0]
            strides = 1 if isinstance(c, int) else c[1]
            layer += [Block(in_channels, out_channels, lambda_l2, strides)]
            in_channels = out_channels
        return tf.keras.Sequential(layer)
