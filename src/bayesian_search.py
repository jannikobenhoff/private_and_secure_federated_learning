import numpy as np
from sklearn.model_selection import KFold
from skopt import gp_minimize, dump, load

from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.utils import use_named_args
from skopt.space import Real
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.models.LeNet import LeNet

if __name__ == "__main__":
    (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()
    img_train = img_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    img_test = img_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        mode='auto',
        restore_best_weights=True,
    )

    search_space = [Real(1e-7, 0.1, "log-uniform", name='lambda_l2')]

    all_acc = []

    @use_named_args(search_space)
    def objective(**params):
        all_scores = []
        model = LeNet(num_classes=10,
                      input_shape=(28, 28, 1),
                      chosen_lambda=None, search=True).search_model(params['lambda_l2'])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        for train_index, val_index in kf.split(img_train):
            x_train, x_val = img_train[train_index], img_train[val_index]
            y_train, y_val = label_train[train_index], label_train[val_index]
            hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=32,
                             verbose=1, callbacks=[early_stopping])

            all_scores.append(hist.history['val_loss'][-1])
        all_acc.append(hist.history['val_accuracy'][-1])
        return np.mean(all_scores)


    result = gp_minimize(objective, search_space, n_calls=5, n_initial_points=0, x0=[0.08], random_state=1)

    print("Best lambda: {}".format(result.x))
    print("Best validation loss: {}".format(result.fun))

    xiter = [x[0] for x in result.x_iters]
    dump(result, 'bayesian_result_02.pkl', store_objective=False)
    dump(all_acc, 'bayesian_acc_02.pkl')

    fig, axs = plt.subplots(3)

    plot_gaussian_process(result, ax=axs[0], show_acq_funcboolean=True)

    axs[1].scatter(xiter, all_acc)

    plot_convergence(result, ax=axs[2])

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    plt.tight_layout()
    plt.show()
