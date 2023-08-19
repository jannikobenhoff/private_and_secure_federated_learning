from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add
from keras.models import Model
import tensorflow as tf
from keras.regularizers import l2
from keras import layers


#
# class ResnetBlock(Model):
#     """
#     A standard resnet block.
#     """
#
#     def __init__(self, lambda_l2, channels: int, down_sample=False):
#         """
#         channels: same as number of convolution kernels
#         """
#         super().__init__()
#
#         self.__channels = channels
#         self.__down_sample = down_sample
#         self.__strides = [2, 1] if down_sample else [1, 1]
#
#         KERNEL_SIZE = (3, 3)
#         # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
#         INIT_SCHEME = "he_normal"
#
#         self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
#                              kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME,
#                              kernel_regularizer=l2(lambda_l2))
#         self.bn_1 = BatchNormalization()
#         self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
#                              kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME,
#                              kernel_regularizer=l2(lambda_l2))
#         self.bn_2 = BatchNormalization()
#         self.merge = Add()
#
#         if self.__down_sample:
#             # perform down sampling using stride of 2, according to [1].
#             self.res_conv = Conv2D(
#                 self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same",
#                 kernel_regularizer=l2(lambda_l2))
#             self.res_bn = BatchNormalization()
#
#     def call(self, inputs, training=None, mask=None):
#         res = inputs
#
#         x = self.conv_1(inputs)
#         x = self.bn_1(x)
#         x = tf.nn.relu(x)
#         x = self.conv_2(x)
#         x = self.bn_2(x)
#
#         if self.__down_sample:
#             res = self.res_conv(res)
#             res = self.res_bn(res)
#
#         # if not perform down sample, then add a shortcut directly
#         x = self.merge([x, res])
#         out = tf.nn.relu(x)
#         return out
#
#
# class ResNet18(Model):
#
#     def __init__(self, num_classes, lambda_l2, **kwargs):
#         """
#             num_classes: number of classes in specific classification task.
#         """
#         super().__init__(**kwargs)
#         self.conv_1 = Conv2D(64, (7, 7), strides=2,
#                              padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(lambda_l2))
#         self.init_bn = BatchNormalization()
#         self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
#         self.res_1_1 = ResnetBlock(lambda_l2, 64)
#         self.res_1_2 = ResnetBlock(lambda_l2, 64)
#         self.res_2_1 = ResnetBlock(lambda_l2, 128, down_sample=True)
#         self.res_2_2 = ResnetBlock(lambda_l2, 128)
#         self.res_3_1 = ResnetBlock(lambda_l2, 256, down_sample=True)
#         self.res_3_2 = ResnetBlock(lambda_l2, 256)
#         self.res_4_1 = ResnetBlock(lambda_l2, 512, down_sample=True)
#         self.res_4_2 = ResnetBlock(lambda_l2, 512)
#         self.avg_pool = GlobalAveragePooling2D()
#         self.flat = Flatten()
#         self.fc = Dense(num_classes, activation="softmax")
#
#     def call(self, inputs, training=None, mask=None):
#         out = self.conv_1(inputs)
#         out = self.init_bn(out)
#         out = tf.nn.relu(out)
#         out = self.pool_2(out)
#         for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
#                           self.res_4_1, self.res_4_2]:
#             out = res_block(out)
#         out = self.avg_pool(out)
#         out = self.flat(out)
#         out = self.fc(out)
#         return out


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
