import tensorflow as tf
from keras.applications import ResNet50, ResNet101
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Add
from keras.regularizers import l2
from keras import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam

class ResNet:
    # @staticmethod
    # def search_model(lambda_l2, input_shape, num_classes):
    #     # Load the ResNet50 model without the top layer (which consists of fully connected layers)
    #     resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    #     output = resnet.output
    #     output = layers.Flatten()(output)
    #     output = layers.Dense(num_classes, activation='softmax')(output)
    #     for layer in resnet.layers:
    #         if hasattr(layer, "kernel_regularizer"):
    #             layer.kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
    #     model = Model(resnet.input, output)
    #     # for layer in model.layers:
    #     #     if hasattr(layer, "kernel_regularizer"):
    #     #         print(layer.kernel_regularizer)
    #     return model

    def search_model(self, lambda_l2, input_shape, num_classes):
        inputs = Input(shape=input_shape)

        x = Conv2D(32, 3, padding='same', kernel_regularizer=l2(lambda_l2))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = self.res_block(x, 32, lambda_l2=lambda_l2)
        x = self.res_block(x, 32, lambda_l2=lambda_l2)

        x = self.res_block(x, 64, lambda_l2=lambda_l2)
        x = self.res_block(x, 64, lambda_l2=lambda_l2)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)  # kernel_regularizer=l2(lambda_l2)

        return Model(inputs, outputs)

    @staticmethod
    def res_block(x, filters, kernel_size=3, lambda_l2=None):
        shortcut = x

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(lambda_l2))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(lambda_l2))(x)
        x = BatchNormalization()(x)

        if x.shape[-1] != shortcut.shape[-1]:
            shortcut = Conv2D(filters, kernel_size=1, padding='same', kernel_regularizer=l2(lambda_l2))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = ReLU()(x)

        return x
