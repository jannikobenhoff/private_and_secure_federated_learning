import numpy as np
import keras
import tensorflow as tf


def load_dataset(dataset_name: str, fullset=100):
    seed_value = 100
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    print(f'==> Preparing {dataset_name.upper()} dataset..')

    img_train, label_train, img_test, label_test, input_shape, num_classes = None, None, None, None, None, None
    if dataset_name == "mnist":
        (img_train_full, label_train_full), (img_test_full, label_test_full) = tf.keras.datasets.mnist.load_data()

        shuffle_indices_train = np.arange(len(img_train_full))
        np.random.shuffle(shuffle_indices_train)

        img_train_full = img_train_full[shuffle_indices_train]
        label_train_full = label_train_full[shuffle_indices_train]

        shuffle_indices_test = np.arange(len(img_test_full))
        np.random.shuffle(shuffle_indices_test)

        img_test_full = img_test_full[shuffle_indices_test]
        label_test_full = label_test_full[shuffle_indices_test]

        train_len = int(len(img_train_full) * (fullset / 100.0))
        test_len = int(len(img_test_full) * (fullset / 100.0))

        img_train = img_train_full[:train_len].reshape(-1, 28, 28, 1).astype('float32') / 255.0
        label_train = label_train_full[:train_len]
        img_test = img_test_full[:test_len].reshape(-1, 28, 28, 1).astype('float32') / 255.0
        label_test = label_test_full[:test_len]

        input_shape = img_train[0].shape
        num_classes = 10
    if dataset_name == "cifar10":
        (img_train_full, label_train_full), (img_test_full, label_test_full) = keras.datasets.cifar10.load_data()

        shuffle_indices_train = np.arange(len(img_train_full))
        np.random.shuffle(shuffle_indices_train)

        img_train_full = img_train_full[shuffle_indices_train]
        label_train_full = label_train_full[shuffle_indices_train]

        shuffle_indices_test = np.arange(len(img_test_full))
        np.random.shuffle(shuffle_indices_test)

        img_test_full = img_test_full[shuffle_indices_test]
        label_test_full = label_test_full[shuffle_indices_test]

        train_len = int(len(img_train_full) * (fullset / 100.0))
        test_len = int(len(img_test_full) * (fullset / 100.0))

        img_train = img_train_full[:train_len].astype('float32') / 255.0
        label_train = label_train_full[:train_len]

        img_test = img_test_full[:test_len].astype('float32') / 255.0
        label_test = label_test_full[:test_len]

        # one hot enoding
        # label_train = to_categorical(label_train)
        # label_test = to_categorical(label_test)

        input_shape = img_train[0].shape
        num_classes = 10

    return img_train, label_train, img_test, label_test, input_shape, num_classes


if __name__ == "__main__":
    print(load_dataset("cifar10"))
