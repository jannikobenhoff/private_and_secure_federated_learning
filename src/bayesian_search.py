import numpy as np
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from skopt.plots import plot_gaussian_process
from keras import datasets, models, layers, optimizers, regularizers
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
import tensorflow as tf

from src.compressions.TopK import TopK
from src.utilities.datasets import load_dataset
from src.utilities.strategy import Strategy

if __name__ == "__main__":
    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("mnist", fullset=10)


    space = [Real(1e-7, 1e-1, "log-uniform", name='l2_reg')]  #

    @use_named_args(space)
    def objective(**params):
        tf.config.set_visible_devices([], 'GPU')
        print("Lambda: ", params["l2_reg"])

        validation_acc_list = []
        val_loss_list = []

        kfold = KFold(n_splits=5, shuffle=True)
        for train_index, val_index in kfold.split(img_train):
            train_images, val_images = img_train[train_index], img_train[val_index] #img_train, img_test  #
            train_labels, val_labels = label_train[train_index], label_train[val_index] #label_train, label_test  #

            model = tf.keras.models.Sequential()
            model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape,kernel_regularizer=regularizers.l2(params["l2_reg"])))
            model.add(layers.MaxPooling2D())
            model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(params["l2_reg"])))
            model.add(layers.MaxPooling2D())
            model.add(layers.Flatten())
            model.add(layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(params["l2_reg"])))
            model.add(layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(params["l2_reg"])))
            model.add(layers.Dense(10, activation='softmax'))

            # def custom_loss(y_true, y_pred):
            #     loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            #     l2_loss = params["l2_reg"] * tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
            #     return loss + l2_loss
            # strategy = Strategy(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1),compression=TopK(k=20))

            model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),  # strategy.optimizer,
                          loss='sparse_categorical_crossentropy',  #custom_loss, #
                          metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1)

            history = model.fit(train_images, train_labels, epochs=250, batch_size=32,
                                validation_data=(val_images, val_labels), verbose=0, callbacks=[early_stopping])

            validation_acc = np.mean(history.history['val_accuracy'])
            val_loss = np.mean(history.history['val_loss'])  # np.min(history.history['val_loss'])  history.history['val_loss'][-1]
            validation_acc_list.append(validation_acc)
            val_loss_list.append(val_loss)

        print('Val Acc: ', np.mean(validation_acc_list))
        print('Val Loss:', np.mean(val_loss_list))

        return - np.mean(validation_acc_list)#np.mean(val_loss_list)  #

    res_gp = gp_minimize(objective, space, n_calls=10, verbose=0, random_state=45, #n_random_starts=3,
                         acq_func='EI', #kappa=1,
                         #x0=[[1e-6], [1e-4], [0.01]],
                         )

    print("Best parameters: ", res_gp.x)
    print("Best validation accuracy: ", res_gp.fun)
    plot_gaussian_process(res_gp, ashow_title=True,
                          show_legend=True, show_next_point=True, show_observations=True)
    plt.xscale('log')
    plt.show()
