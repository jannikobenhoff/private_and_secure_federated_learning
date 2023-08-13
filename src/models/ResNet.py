from keras import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Layer, \
    AveragePooling2D, Flatten
from keras.models import Model
from keras.regularizers import l2


def residual_block(x, filters, kernel_size=3, stride=1, regularization_factor=0.001):
    # Shortcut
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same",
               kernel_regularizer=l2(regularization_factor))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same",
               kernel_regularizer=l2(regularization_factor))(x)
    x = BatchNormalization()(x)

    # Adding the shortcut to the output
    shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding="same",
                      kernel_regularizer=l2(regularization_factor))(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x


def resnet18(input_shape, num_classes, regularization_factor=None):
    # Input layer
    input = Input(input_shape)

    # Initial Conv Layer
    x = Conv2D(64, kernel_size=7, strides=2, padding="same", kernel_regularizer=l2(regularization_factor))(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual Blocks
    x = residual_block(x, 64, regularization_factor=regularization_factor)
    x = residual_block(x, 64, regularization_factor=regularization_factor)

    x = residual_block(x, 128, regularization_factor=regularization_factor)
    x = residual_block(x, 128, regularization_factor=regularization_factor)

    x = residual_block(x, 256, regularization_factor=regularization_factor)
    x = residual_block(x, 256, regularization_factor=regularization_factor)

    x = residual_block(x, 512, regularization_factor=regularization_factor)
    x = residual_block(x, 512, regularization_factor=regularization_factor)

    # Global Average Pooling Layer
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer (Dense Layer)
    output = Dense(num_classes, activation="softmax")(x)

    # Create Model
    model = Model(inputs=input, outputs=output)

    return model


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add(Conv2D(self.expansion * planes, kernel_size=1, strides=stride, use_bias=False))
            self.shortcut.add(BatchNormalization())

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)

        out = Add()([out, shortcut])
        out = ReLU()(out)

        return out


def ResNet18(num_classes=10):
    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = _make_layer(x, BasicBlock, 64, 2, stride=1)
    x = _make_layer(x, BasicBlock, 128, 2, stride=2)
    x = _make_layer(x, BasicBlock, 256, 2, stride=2)
    x = _make_layer(x, BasicBlock, 512, 2, stride=2)

    x = AveragePooling2D(4)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)

    return model


def _make_layer(x, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    in_planes = x.shape[-1]

    for stride in strides:
        x = block(in_planes, planes, stride)(x)
        in_planes = planes * block.expansion

    return x


from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, lambda_l2, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME,
                             kernel_regularizer=l2(lambda_l2))
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME,
                             kernel_regularizer=l2(lambda_l2))
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same",
                kernel_regularizer=l2(lambda_l2))
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18_new(Model):

    def __init__(self, num_classes, lambda_l2, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(lambda_l2))
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(lambda_l2, 64)
        self.res_1_2 = ResnetBlock(lambda_l2, 64)
        self.res_2_1 = ResnetBlock(lambda_l2, 128, down_sample=True)
        self.res_2_2 = ResnetBlock(lambda_l2, 128)
        self.res_3_1 = ResnetBlock(lambda_l2, 256, down_sample=True)
        self.res_3_2 = ResnetBlock(lambda_l2, 256)
        self.res_4_1 = ResnetBlock(lambda_l2, 512, down_sample=True)
        self.res_4_2 = ResnetBlock(lambda_l2, 512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
                          self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

# class ResNet:
#     # @staticmethod
#     # def search_model(lambda_l2, input_shape, num_classes):
#     #     # Load the ResNet50 model without the top layer (which consists of fully connected layers)
#     #     resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#     #     output = resnet.output
#     #     output = layers.Flatten()(output)
#     #     output = layers.Dense(num_classes, activation='softmax')(output)
#     #     for layer in resnet.layers:
#     #         if hasattr(layer, "kernel_regularizer"):
#     #             layer.kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
#     #     model = Model(resnet.input, output)
#     #     # for layer in model.layers:
#     #     #     if hasattr(layer, "kernel_regularizer"):
#     #     #         print(layer.kernel_regularizer)
#     #     return model
#
#     def search_model(self, lambda_l2, input_shape, num_classes):
#         inputs = Input(shape=input_shape)
#
#         x = Conv2D(32, 3, padding='same', kernel_regularizer=l2(lambda_l2))(inputs)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#
#         x = self.res_block(x, 32, lambda_l2=lambda_l2)
#         x = self.res_block(x, 32, lambda_l2=lambda_l2)
#
#         x = self.res_block(x, 64, lambda_l2=lambda_l2)
#         x = self.res_block(x, 64, lambda_l2=lambda_l2)
#
#         x = GlobalAveragePooling2D()(x)
#         outputs = Dense(num_classes, activation='softmax')(x)  # kernel_regularizer=l2(lambda_l2)
#
#         return Model(inputs, outputs)
#
#     @staticmethod
#     def res_block(x, filters, kernel_size=3, lambda_l2=None):
#         shortcut = x
#
#         x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(lambda_l2))(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#
#         x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(lambda_l2))(x)
#         x = BatchNormalization()(x)
#
#         if x.shape[-1] != shortcut.shape[-1]:
#             shortcut = Conv2D(filters, kernel_size=1, padding='same', kernel_regularizer=l2(lambda_l2))(shortcut)
#             shortcut = BatchNormalization()(shortcut)
#
#         x = Add()([x, shortcut])
#         x = ReLU()(x)
#
#         return x
