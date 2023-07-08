from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import backend as K

from models.LeNet import LeNet


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":
    logdir = "results/logs/scalars/baseline_model-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        data_dir='datasets/mnist',
        download=True,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    input_size = ds_info.features['image'].shape
    print("Input size:", input_size)

    baselineModel = LeNet(input_shape=input_size)
    baselineModel.summary()

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    baselineModel.compile(
        optimizer=tf.keras.optimizers.SGD(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    baselineModel.fit(ds_train, epochs=10, validation_data=ds_test,
                      callbacks=[tensorboard_callback],
                      )

