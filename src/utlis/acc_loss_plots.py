import pandas as pd
import matplotlib.pyplot as plt
import os

# use read_logs.py first
root_path = "../logs/experiment4_batch_size/rs25/"
# change the folder path to the folder which contains generated log folders only
folder_names = os.listdir(root_path)

for folder_name in folder_names:
    if folder_name.startswith('final_'):
        continue
    path = root_path + folder_name + '/'

    train_acc = pd.read_csv(path + "acc/train_clients_average.csv")
    #print(train_acc.head())
    train_federator_acc = pd.read_csv(path + "acc/train_clients_average_federator.csv")
    #val_acc = pd.read_csv(path + "acc/validation_global.csv")
    test_acc = pd.read_csv(path + "acc/test.csv")

    train_loss = pd.read_csv(path + "loss/train_clients_average.csv")
    train_federator_loss = pd.read_csv(path + "loss/train_clients_average_federator.csv")
    test_loss = pd.read_csv(path + "loss/test.csv")

    plt.figure(1, figsize=(8, 8))
    plt.loglog(train_loss.iloc[:, 1], train_loss.iloc[:, 2], label='train')
    plt.loglog(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], label='train after updating')
    plt.loglog(test_loss.iloc[:, 1], test_loss.iloc[:, 2], label='validation')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch Number (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.grid(visible=True)
    plt.savefig(path + 'Loss vs Epoch (log scale).png')

    plt.figure(2, figsize=(8, 8))
    plt.semilogx(train_acc.iloc[:, 1], train_acc.iloc[:, 2], label='train')
    plt.semilogx(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], label='train after updating')
    plt.semilogx(test_acc.iloc[:, 1], test_acc.iloc[:, 2], label='validation')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch Number (log scale)')
    plt.ylabel('Accuracy')
    plt.grid(visible=True)
    plt.savefig(path + 'Accuracy vs Epoch (log scale).png')

    plt.figure(3, figsize=(8, 8))
    plt.plot(train_loss.iloc[:, 1], train_loss.iloc[:, 2], label='train')
    plt.plot(train_federator_loss.iloc[1:, 1], train_federator_loss.iloc[1:, 2], label='train after updating')
    plt.plot(test_loss.iloc[:, 1], test_loss.iloc[:, 2], label='validation')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.grid(visible=True)
    plt.savefig(path + 'Loss vs Epoch.png')

    plt.figure(4, figsize=(8, 8))
    plt.plot(train_acc.iloc[:, 1], train_acc.iloc[:, 2], label='train')
    plt.plot(train_federator_acc.iloc[1:, 1], train_federator_acc.iloc[1:, 2], label='train after updating')
    plt.plot(test_acc.iloc[:, 1], test_acc.iloc[:, 2], label='validation')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.grid(visible=True)
    plt.savefig(path + 'Accuracy vs Epoch.png')

    plt.close('all')
