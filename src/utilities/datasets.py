from tensorflow import keras
import tensorflow as tf

def load_dataset(dataset_name: str):
    print(f"Loading {dataset_name} dataset.")
    img_train, label_train, img_test, label_test, input_shape, num_classes = None, None, None, None, None, None
    if dataset_name == "mnist":
        (img_train, label_train), (img_test, label_test) = keras.datasets.mnist.load_data()
        img_train = img_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        img_test = img_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        input_shape = img_train[0].shape
        num_classes = 10
    if dataset_name == "cifar10":
        (img_train, label_train), (img_test, label_test) = keras.datasets.cifar10.load_data()
        img_train = img_train.astype('float32') / 255.0
        img_test = img_test.astype('float32') / 255.0
        input_shape = img_train[0].shape
        num_classes = 10

    if dataset_name == "cifar100":
        (img_train, label_train), (img_test, label_test) = keras.datasets.cifar100.load_data()
        img_train = img_train.astype('float32') / 255.0
        img_test = img_test.astype('float32') / 255.0
        input_shape = img_train[0].shape
        num_classes = 100

    return img_train, label_train, img_test, label_test, input_shape, num_classes


if __name__ == "__main__":
    print(load_dataset("cifar100"))
