from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from keras_tuner import Objective
from tensorflow import keras
import os
from tensorboard import program
from keras_tuner.tuners import BayesianOptimization
from models.LeNet import LeNet
from sklearn.model_selection import KFold


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":
    logdir = "results/logs/scalars/baseline_model-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()
    img_train = img_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    img_test = img_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    label_train = keras.utils.to_categorical(label_train, num_classes)
    label_test = keras.utils.to_categorical(label_test, num_classes)

    baselineModel = LeNet()

    early_stopping = tf.keras.callbacks.EarlyStopping(
                                     monitor='val_loss',
                                     min_delta=0,
                                     patience=3,
                                     mode='auto',
                                     baseline=None,
                                     restore_best_weights=False,
                                     start_from_epoch=0
                                 )

    kf = KFold(n_splits=5, shuffle=True)

    tuner = BayesianOptimization(
        baselineModel.build_model,
        objective="val_accuracy",
        max_trials=10,
        directory='results/tuner_results',
        project_name='l2_regularization_7'
    )

    for train_index, val_index in kf.split(img_train):
        x_train, x_val = img_train[train_index], img_train[val_index]
        y_train, y_val = label_train[train_index], label_train[val_index]
        tuner.search(x_train, y_train,
                     validation_data=(x_val, y_val),
                     batch_size=32,
                     epochs=50,
                     callbacks=[early_stopping, tensorboard_callback])
        # tuner.search(img_train, label_train,
        #              batch_size=32,
        #              epochs=10,
        #              validation_split=0.2,
        #              callbacks=[early_stopping, tensorboard_callback]
        #              )

    # Get the best lambda value
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_lambda = best_hp.get('lambda')
    print("Best Lambda:", best_lambda)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', os.path.abspath(logdir)])
    tb.main()

    """
    Best lambdas so far:
    - 0.08242441230782568 - 25 iterations - 1e-4, 0.1 - 10 epochs
    - 0.05160594941471645 - 25 iterations - 0.05, 0.1 - 10 epochs
    - 0.01527577498554062 - 25 iterations - 0.01, 0.07 - 10 epochs
    - 0.01527577498554062 - 25 iterations - 0.01, 0.07 - 10 epochs
    - 0.0022699912399919103 - 20 iterations - 0.001, 0.1 - 25 epochs
    - 0.0018463634346080097 - 10 iterations - 0.001, 0.1 - 50 epochs
    """
