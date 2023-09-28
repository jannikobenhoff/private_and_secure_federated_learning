import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def label_splitter(data, labels):
    unique_labels = np.unique(labels)
    number_labels = unique_labels.size
    list_data = list()
    list_labels = list()

    for k in range(number_labels):
        indices = np.where(labels == unique_labels[k])
        list_data.append(data[indices[0], :, :, :])
        list_labels.append(labels[indices[0]])
    return list_data, list_labels


def stratified_sampling(num_sample_per_class: int, list_data: list, list_labels: list):
    sample_data = list()
    sample_labels = list()
    for i in range(len(list_labels)):
        data = list_data[i]
        label = list_labels[i]
        sample_indices = np.random.choice(len(list_labels[i]), size=num_sample_per_class, replace=False)
        sample_data.append(data[sample_indices, :, :, :])
        sample_labels.append(label[sample_indices])
    return sample_data, sample_labels


def client_datasets(number_clients: int, split_type: str, list_data: list, list_labels: list, beta, dataset="mnist"):
    number_label = len(list_labels)
    list_client_label = list()
    list_client_data = list()
    split_data_array = list()
    split_label_array = list()
    # image size
    height = len(list_data[0][0])
    weight = len(list_data[0][0][0])
    channel = len(list_data[0][0][0][0])
    # print(height, weight, channel)

    if split_type == 'class-clients':  # Each client has exactly one label
        list_client_data = list_data
        list_client_label = list_labels

    elif split_type == 'half-class-clients':  # Each client is dominated by one label
        for m in range(number_label):
            # split_data_array.append( np.vsplit(list_data[m], number_label*2) )
            # split_label_array.append( np.vsplit(list_labels[m], number_label*2) )
            split_data_array.append(np.array_split(list_data[m], number_clients * 2))
            split_label_array.append(np.array_split(list_labels[m], number_clients * 2))

        for k in range(number_label):
            data = np.empty(shape=[1, height, weight, channel], dtype=int)
            if dataset == "cifar10":
                label = np.empty(shape=[1, 1], dtype=int)
            else:
                label = np.empty(shape=[1], dtype=int)

            for n in range(number_label):
                if k == n:
                    for p in range(number_label + 1):
                        data = np.concatenate((data, split_data_array[n][p]), axis=0)
                        label = np.concatenate((label, split_label_array[n][p]), axis=0)

                else:
                    data = np.concatenate((data, split_data_array[n][number_label + k]), axis=0)
                    label = np.concatenate((label, split_label_array[n][number_label + k]), axis=0)

            data = np.delete(data, (0), axis=0)
            label = np.delete(label, (0), axis=0)

            list_client_data.append(data)
            list_client_label.append(label)

    elif split_type == 'uniform':  # Each client has uniformly distributed labels
        for m in range(number_label):
            # split_data_array.append( np.vsplit(list_data[m], number_clients))
            # split_label_array.append( np.vsplit(list_labels[m], number_clients))
            # numbers of images in each class of mnist are not equal!
            split_data_array.append(np.array_split(list_data[m], number_clients))
            split_label_array.append(np.array_split(list_labels[m], number_clients))

        for k in range(number_clients):
            data = np.empty(shape=[1, height, weight, channel], dtype=int)
            if dataset == "cifar10":
                label = np.empty(shape=[1, 1], dtype=int)
            else:
                label = np.empty(shape=[1], dtype=int)

            for n in range(number_label):
                data = np.concatenate((data, split_data_array[n][k]), axis=0)
                label = np.concatenate((label, split_label_array[n][k]), axis=0)

            data = np.delete(data, (0), axis=0)
            label = np.delete(label, (0), axis=0)

            list_client_data.append(data)
            list_client_label.append(label)

    elif split_type == 'random':  # Each client has randomly distributed labels
        for m in range(number_label):
            split_points = np.sort(np.random.randint(0, len(list_labels[m]), number_clients - 1))
            # split_data_array.append(np.vsplit(list_data[m], split_points))
            # split_label_array.append(np.vsplit(list_labels[m], split_points))
            split_data_array.append(np.array_split(list_data[m], split_points))
            split_label_array.append(np.array_split(list_labels[m], split_points))

        for k in range(number_clients):
            data = np.empty(shape=[1, height, weight, channel], dtype=int)
            if dataset == "cifar10":
                label = np.empty(shape=[1, 1], dtype=int)
            else:
                label = np.empty(shape=[1], dtype=int)

            for n in range(number_label):
                data = np.concatenate((data, split_data_array[n][k]), axis=0)
                label = np.concatenate((label, split_label_array[n][k]), axis=0)

            data = np.delete(data, (0), axis=0)
            label = np.delete(label, (0), axis=0)

            list_client_data.append(data)
            list_client_label.append(label)

    elif split_type == 'dirichlet':
        sample_rate = np.random.dirichlet([beta] * number_clients, size=number_label)
        # print(sample_rate)

        for n in range(number_label):
            split_points = [0] * (number_clients - 1)
            for k in range(number_clients - 1):
                num_sample = int(sample_rate[n, k] * len(list_labels[n]))
                split_points[k] = num_sample + split_points[k - 1]
            split_data_array.append(np.array_split(list_data[n], split_points))
            split_label_array.append(np.array_split(list_labels[n], split_points))

        for k in range(number_clients):
            data = np.empty(shape=[1, height, weight, channel], dtype=int)
            if dataset == "cifar10":
                label = np.empty(shape=[1, 1], dtype=int)
            else:
                label = np.empty(shape=[1], dtype=int)

            for n in range(number_label):
                data = np.concatenate((data, split_data_array[n][k]), axis=0)
                label = np.concatenate((label, split_label_array[n][k]), axis=0)

            data = np.delete(data, (0), axis=0)
            label = np.delete(label, (0), axis=0)

            list_client_data.append(data)
            list_client_label.append(label)

    else:
        print('Not Defined Split Type')

    for k in range(len(list_client_label)):
        randomize = np.arange(len(list_client_label[k]))
        np.random.shuffle(randomize)
        list_client_data[k] = list_client_data[k][randomize]
        list_client_label[k] = list_client_label[k][randomize]

    return list_client_data, list_client_label
