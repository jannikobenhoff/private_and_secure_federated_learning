from keras import regularizers
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# def alexnet(input_shape=(227, 227, 3), num_classes=1000):
#     model = Sequential([
#         # 1st Conv Layer
#         Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding='valid'),
#         MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
#         tf.keras.layers.BatchNormalization(),
#
#         # 2nd Conv Layer
#         Conv2D(256, (5, 5), activation='relu', padding='same'),
#         MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
#         tf.keras.layers.BatchNormalization(),
#
#         # 3rd Conv Layer
#         Conv2D(384, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#
#         # 4th Conv Layer
#         Conv2D(384, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#
#         # 5th Conv Layer
#         Conv2D(256, (3, 3), activation='relu', padding='same'),
#         MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
#         tf.keras.layers.BatchNormalization(),
#
#         Flatten(),
#
#         # 1st Fully Connected Layer
#         Dense(4096, activation='relu'),
#         Dropout(0.5),
#
#         # 2nd Fully Connected Layer
#         Dense(4096, activation='relu'),
#         Dropout(0.5),
#
#         # 3rd Fully Connected Layer (Output Layer)
#         Dense(num_classes, activation='softmax')
#     ])
#
#     return model

def alexnet(lambda_l2, input_shape):
    model = keras.models.Sequential([
        Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=input_shape, kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same", kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same", kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same", kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same", kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(lambda_l2)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')

    ])

    return model

def AlexnetModel(input_shape,num_classes):
  model = Sequential()
  model.add(Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=input_shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(256,(5,5),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(num_classes,activation='softmax'))

  #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

  #model.summary()
  return model