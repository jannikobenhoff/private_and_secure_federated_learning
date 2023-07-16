from datetime import datetime

import tensorflow as tf
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

    print("Training on", len(img_train), "images")
    baselineModel = LeNet()

    early_stopping = keras.callbacks.EarlyStopping(
                                     monitor='val_loss',
                                     patience=7,
                                     mode='auto',
                                     restore_best_weights=True,
                                 )

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, val_index in kf.split(img_train):
        x_train, x_val = img_train[train_index], img_train[val_index]
        y_train, y_val = label_train[train_index], label_train[val_index]

        tuner = BayesianOptimization(
            baselineModel.build_model,
            objective="val_accuracy",
            max_trials=125,
            directory='results/tuner_results',
            project_name='l2_regularization_11'
        )

        tuner.search(x_train, y_train,
                     validation_data=(x_val, y_val),
                     batch_size=32,
                     epochs=40,
                     callbacks=[early_stopping, tensorboard_callback])

    # Get the best lambda value
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_lambda = best_hp.get('lambda')
    print("Best Lambda:", best_lambda)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', os.path.abspath(logdir)])
    tb.main()

    """
    Best lambdas so far:
    Lambda: 0.0016932133792534388 - val_accuracy: 0.9819166660308838 - iter: 50, epochs: 30 - 1e-6 - 0.1
    Lambda: 0.0017854750244078188 - val_accuracy: 0.9819999933242798 - iter: 100, epochs: 30 - 1e-6 - 0.01
    Lambda: 0.0001952415460342464 - val_accuracy: 0.9900000095367432 - iter: 100, epochs: 25 - 1e-6 - 0.1
    Lambda: 2.001382315994955e-06 - val_accuracy: 0.9900000095367432 - iter: 125, epochs: 25 - 1e-6 - 0.001
    """
