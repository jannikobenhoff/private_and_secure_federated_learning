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
from src.models.ResNet import ResNet
from src.optimizer.FetchSGD import FetchSGD
from src.optimizer.SGD import SGD
from src.utilities.datasets import load_dataset
from src.utilities.strategy import Strategy

if __name__ == "__main__":
    img_train, label_train, img_test, label_test, input_shape, num_classes = load_dataset("cifar10")
    n_splits = 2
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
        strategy = Strategy(optimizer=SGD(learning_rate=0.1))

        all_scores = []
        # model = LeNet(num_classes=10,
        #               input_shape=(28, 28, 1),
        #               chosen_lambda=None, search=True).search_model(params['lambda_l2'])
        model = ResNet.search_model(params['lambda_l2'], input_shape=input_shape, num_classes=num_classes)
        model.compile(optimizer=strategy.optimizer, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        for train_index, val_index in kf.split(img_train):
            x_train, x_val = img_train[train_index], img_train[val_index]
            y_train, y_val = label_train[train_index], label_train[val_index]
            hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=128,
                             verbose=1, callbacks=[early_stopping])

            all_scores.append(hist.history['val_loss'][-1])
        all_acc.append(hist.history['val_accuracy'][-1])
        return np.mean(all_scores)


    result = gp_minimize(objective, search_space, n_calls=5, n_initial_points=0, x0=[0.08], random_state=1)

    print("Best lambda: {}".format(result.x))
    print("Best validation loss: {}".format(result.fun))

    # metrics = {
    #     "training_loss": training_losses_per_epoch,
    #     "training_acc": training_acc_per_epoch,
    #     "val_loss": validation_losses_per_epoch,
    #     "val_acc": validation_acc_per_epoch,
    #     "args": args
    # }
    # result["metrics"] = metrics
    dump(result, f'results/bayesian/bayesian_result_SGD_Cifar.pkl', store_objective=False)
    # print(f"Finished search for {strategy.get_plot_title()}")
