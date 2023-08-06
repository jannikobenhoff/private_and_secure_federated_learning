import tensorflow as tf
from keras.applications import ResNet50, ResNet101
from keras import layers, models, regularizers
from keras.models import Model


class ResNet:
    @staticmethod
    def search_model(lambda_l2, input_shape, num_classes):
        # Load the ResNet50 model without the top layer (which consists of fully connected layers)
        resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        output = resnet.output
        output = layers.Flatten()(output)
        output = layers.Dense(num_classes, activation='softmax')(output)
        for layer in resnet.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
        model = Model(resnet.input, output)
        # for layer in model.layers:
        #     if hasattr(layer, "kernel_regularizer"):
        #         print(layer.kernel_regularizer)
        return model

