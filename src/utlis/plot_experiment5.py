import pandas as pd
import matplotlib.pyplot as plt
import os

root_path = "../logs/experiment5_local_iter_varying/plots/dirichlet_gaussian/"
folder_names = os.listdir(root_path)

plt.figure(1, figsize=(8, 8))
for folder_name in folder_names:
    if folder_name.startswith('Accuracy') or folder_name.startswith('Loss'):
        continue
    path = root_path + folder_name + '/'
    train_loss = pd.read_csv(path + "loss/train_clients_average.csv")
    train_federator_loss = pd.read_csv(path + "loss/train_clients_average_federator.csv")
    test_loss = pd.read_csv(path + "loss/test.csv")
    if folder_name == 'base line':
        plt.plot(train_loss.iloc[:, 1], train_loss.iloc[:, 2], '--', label='train (' + folder_name + ')')
        plt.plot(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], '--',
                 label='train after updating (' + folder_name + ')')
        plt.plot(test_loss.iloc[:, 1], test_loss.iloc[:, 2], '--', label='validation (' + folder_name + ')')
    else:
        plt.plot(train_loss.iloc[:, 1], train_loss.iloc[:, 2], label='train ('+folder_name+')')
        plt.plot(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], label='train after updating ('+folder_name+')')
        plt.plot(test_loss.iloc[:, 1], test_loss.iloc[:, 2], label='validation ('+folder_name+')')
plt.legend()
plt.title('Loss vs Epoch')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.grid(visible=True)
plt.savefig(root_path + 'Loss vs Epoch.png')


plt.figure(2, figsize=(8, 8))
for folder_name in folder_names:
    if folder_name.startswith('Accuracy') or folder_name.startswith('Loss'):
        continue
    path = root_path + folder_name + '/'
    train_acc = pd.read_csv(path + "acc/train_clients_average.csv")
    train_federator_acc = pd.read_csv(path + "acc/train_clients_average_federator.csv")
    test_acc = pd.read_csv(path + "acc/test.csv")

    train_loss = pd.read_csv(path + "loss/train_clients_average.csv")
    train_federator_loss = pd.read_csv(path + "loss/train_clients_average_federator.csv")
    test_loss = pd.read_csv(path + "loss/test.csv")
    if folder_name == 'base line':
        plt.plot(train_acc.iloc[:, 1], train_acc.iloc[:, 2], '--', label='train (' + folder_name + ')')
        plt.plot(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], '--',
                 label='train after updating  (' + folder_name + ')')
        plt.plot(test_acc.iloc[:, 1], test_acc.iloc[:, 2], '--', label='validation (' + folder_name + ')')
    else:
        plt.plot(train_acc.iloc[:, 1], train_acc.iloc[:, 2], label='train ('+folder_name+')')
        plt.plot(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], label='train after updating  ('+folder_name+')')
        plt.plot(test_acc.iloc[:, 1], test_acc.iloc[:, 2], label='validation ('+folder_name+')')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.grid(visible=True)
plt.savefig(root_path + 'Accuracy vs Epoch.png')

plt.figure(3, figsize=(8, 8))
for folder_name in folder_names:
    if folder_name.startswith('Accuracy') or folder_name.startswith('Loss'):
        continue
    path = root_path + folder_name + '/'
    train_loss = pd.read_csv(path + "loss/train_clients_average.csv")
    train_federator_loss = pd.read_csv(path + "loss/train_clients_average_federator.csv")
    test_loss = pd.read_csv(path + "loss/test.csv")
    if folder_name == 'base line':
        plt.loglog(train_loss.iloc[:, 1], train_loss.iloc[:, 2], '--', label='train (' + folder_name + ')')
        plt.loglog(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], '--',
                   label='train after updating (' + folder_name + ')')
        plt.loglog(test_loss.iloc[:, 1], test_loss.iloc[:, 2], '--', label='validation (' + folder_name + ')')
    else:
        plt.loglog(train_loss.iloc[:, 1], train_loss.iloc[:, 2], label='train ('+folder_name+')')
        plt.loglog(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], label='train after updating ('+folder_name+')')
        plt.loglog(test_loss.iloc[:, 1], test_loss.iloc[:, 2], label='validation ('+folder_name+')')
plt.legend()
plt.title('Loss vs Epoch (log scale)')
plt.xlabel('Epoch Number (log scale)')
plt.ylabel('Loss (log scale)')
plt.grid(visible=True)
plt.savefig(root_path + 'Loss vs Epoch (log scale).png')

plt.figure(4, figsize=(8, 8))
for folder_name in folder_names:
    if folder_name.startswith('Accuracy') or folder_name.startswith('Loss'):
        continue
    path = root_path + folder_name + '/'
    train_acc = pd.read_csv(path + "acc/train_clients_average.csv")
    train_federator_acc = pd.read_csv(path + "acc/train_clients_average_federator.csv")
    test_acc = pd.read_csv(path + "acc/test.csv")
    if folder_name == 'base line':
        plt.semilogx(train_acc.iloc[:, 1], train_acc.iloc[:, 2], '--', label='train (' + folder_name + ')')
        plt.semilogx(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], '--',
                     label='train after updating (' + folder_name + ')')
        plt.semilogx(test_acc.iloc[:, 1], test_acc.iloc[:, 2], '--', label='validation (' + folder_name + ')')
    else:
        plt.semilogx(train_acc.iloc[:, 1], train_acc.iloc[:, 2], label='train ('+folder_name+')')
        plt.semilogx(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], label='train after updating ('+folder_name+')')
        plt.semilogx(test_acc.iloc[:, 1], test_acc.iloc[:, 2], label='validation ('+folder_name+')')
plt.legend()
plt.title('Accuracy vs Epoch (log scale)')
plt.xlabel('Epoch Number (log scale)')
plt.ylabel('Accuracy')
plt.grid(visible=True)
plt.savefig(root_path + 'Accuracy vs Epoch (log scale).png')

plt.close('all')
