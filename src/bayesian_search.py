import os
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from keras.utils import get_custom_objects

from models.LeNet import LeNet
from src.compressions.TopK import TopK
from src.models.ResNet import ResNet
from utilities.datasets import load_dataset
from strategy import Strategy

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    tf.config.set_visible_devices([], 'GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.config.run_functions_eagerly(run_eagerly=True)
    tf.data.experimental.enable_debug_mode()

    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=10)  # 10
    # get_custom_objects().update({"strategy": Strategy})

    # space = [Real(1e-5, 1e-1, "log-uniform", name='l2_reg')]
    space = [Real(1, 200, "uniform", name='l2_reg')]

    @use_named_args(space)
    def objective(**params):
        params["l2_reg"] = int(params["l2_reg"])
        print("Lambda: ", params["l2_reg"])
        validation_acc_list = []
        val_loss_list = []

        kfold = KFold(n_splits=5, shuffle=True)
        for train_index, val_index in kfold.split(img_train):
            train_images, val_images = img_train[train_index], img_train[val_index]
            train_labels, val_labels = label_train[train_index], label_train[val_index]

            # model = ResNet().search_model(lambda_l2=params["l2_reg"], input_shape=input_shape, num_classes=num_classes)
            model = LeNet(search=True).search_model(lambda_l2=None)

            opt = Strategy(compression=TopK(k=params["l2_reg"]), learning_rate=0.01)
            model.compile(optimizer=opt,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

            history = model.fit(train_images, train_labels, epochs=50, batch_size=32,
                                validation_data=(val_images, val_labels), verbose=2, callbacks=[early_stopping])

            validation_acc = np.mean(history.history['val_accuracy'])
            val_loss = np.mean(history.history['val_loss'])
            validation_acc_list.append(validation_acc)
            val_loss_list.append(val_loss)

        print('Val Acc: ', np.mean(validation_acc_list))
        print('Val Loss:', np.mean(val_loss_list))

        return - np.mean(validation_acc_list)


    res_gp = gp_minimize(objective, space, n_calls=10, verbose=0, random_state=45,  # n_random_starts=3,
                         acq_func='EI',  # kappa=1,
                         # x0=[[1e-6], [1e-4], [0.01]],
                         )

    print("Best parameters: ", res_gp.x)
    print("Best validation accuracy: ", res_gp.fun)
    plot_gaussian_process(res_gp, ashow_title=True,
                          show_legend=True, show_next_point=True, show_observations=True)
    plt.xscale('log')
    plt.show()
