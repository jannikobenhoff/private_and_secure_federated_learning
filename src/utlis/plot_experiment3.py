import pandas as pd
import matplotlib.pyplot as plt
import os

path = "../logs/experiment3_same_training_steps/"
file_names_acc = os.listdir(path+'final_acc/')
file_names_loss = os.listdir(path+'final_loss/')
#print(file_names_acc)
final_acc_1 = pd.read_csv(path + 'final_acc/' + file_names_acc[0])
final_loss_1 = pd.read_csv(path + "final_loss/" + file_names_loss[0])
final_acc_2 = pd.read_csv(path + 'final_acc/' + file_names_acc[1])
final_loss_2 = pd.read_csv(path + "final_loss/" + file_names_loss[1])


plt.figure(1, figsize=(10, 10))

plt.loglog(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 2], 'o-', label='train (dirichlet0.5)')
#plt.loglog(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 3], 'o-', label='train after updating (dirichlet0.5)')
plt.loglog(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 1], 'o-', label='validation (dirichlet0.5)')

plt.loglog(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 2], '--', label='train (uniform)')
#plt.loglog(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 3], '--', label='train after updating (uniform)')
plt.loglog(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 1], '--', label='validation (uniform)')

plt.legend()
plt.title('final loss vs local iteration (1000 total training steps)')
plt.xlabel('number of local iterations (log scale)')
plt.ylabel('final loss (log scale)')
plt.grid(visible=True)
plt.savefig(path + 'Final Loss vs Local Iteration log scale (1000 total training steps).png')

plt.figure(2, figsize=(10, 10))
plt.semilogx(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 2], 'o-', label='train (dirichlet0.5)')
#plt.semilogx(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 3], 'o-', label='train after updating (dirichlet0.5)')
plt.semilogx(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 1], 'o-', label='validation (dirichlet0.5)')

plt.semilogx(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 2], '--', label='train (uniform)')
#plt.semilogx(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 3], '--', label='train after updating (uniform)')
plt.semilogx(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 1], '--', label='validation (dirichlet0.5)')

plt.legend()
plt.title('final accuracy vs local iteration (1000 total training steps)')
plt.xlabel('number of local iterations (log scale)')
plt.ylabel('final acc')
plt.grid(visible=True)
plt.savefig(path + 'Final Accuracy vs Local Iteration log scale (1000 total training steps).png')

plt.figure(3, figsize=(10, 10))

plt.plot(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 2], 'o-', label='train (dirichlet0.5)')
#plt.plot(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 3], 'o-', label='train after updating (dirichlet0.5)')
plt.plot(final_loss_1.iloc[:, -1], final_loss_1.iloc[:, 1], 'o-', label='validation (dirichlet0.5)')

plt.plot(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 2], '--', label='train (uniform)')
#plt.plot(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 3], '--', label='train after updating (uniform)')
plt.plot(final_loss_2.iloc[:, -1], final_loss_2.iloc[:, 1], '--', label='validation (uniform)')

plt.legend()
plt.title('final loss vs local iteration (1000 total training steps)')
plt.xlabel('number of local iterations')
plt.ylabel('final loss')
plt.grid(visible=True)
plt.savefig(path + 'Final Loss vs Local Iteration (1000 total training steps).png')

plt.figure(4, figsize=(10, 10))
plt.plot(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 2], 'o-', label='train (dirichlet0.5)')
#plt.plot(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 3], 'o-', label='train after updating (dirichlet0.5)')
plt.plot(final_acc_1.iloc[:, -1], final_acc_1.iloc[:, 1], 'o-', label='validation (dirichlet0.5)')

plt.plot(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 2], '--', label='train (uniform)')
#plt.plot(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 3], '--', label='train after updating (uniform)')
plt.plot(final_acc_2.iloc[:, -1], final_acc_2.iloc[:, 1], '--', label='validation (dirichlet0.5)')

plt.legend()
plt.title('final accuracy vs local iteration (1000 total training steps)')
plt.xlabel('number of local iterations')
plt.ylabel('final acc')
plt.grid(visible=True)
plt.savefig(path + 'Final Accuracy vs Local Iteration (1000 total training steps).png')





