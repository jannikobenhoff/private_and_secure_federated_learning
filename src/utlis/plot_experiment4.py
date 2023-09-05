import pandas as pd
import matplotlib.pyplot as plt

path = "../logs/experiment4_batch_size/"
final_acc = pd.read_csv(path + "final_acc.csv")
final_loss = pd.read_csv(path + "final_loss.csv")


plt.figure(1, figsize=(10, 10))
plt.plot(final_loss.iloc[:, -1], final_loss.iloc[:, 2], 'o-', label='train')
plt.plot(final_loss.iloc[:, -1], final_loss.iloc[:, 3], 'o-', label='train after updating')
plt.plot(final_loss.iloc[:, -1], final_loss.iloc[:, 1], 'o-', label='validation')

plt.legend()
plt.title('final Loss vs batch size')
plt.xlabel('batch size')
plt.ylabel('final Loss')
#plt.ylim([0, 0.6])
plt.grid(visible=True)
plt.savefig(path + 'Final Loss vs Batch Size.png')
#plt.savefig(path + 'Final Loss vs Batch Size (adjusted).png')

plt.figure(2, figsize=(10, 10))
plt.plot(final_acc.iloc[:, -1], final_acc.iloc[:, 2], 'o-', label='train')
plt.plot(final_acc.iloc[:, -1], final_acc.iloc[:, 3], 'o-', label='train after updating')
plt.plot(final_acc.iloc[:, -1], final_acc.iloc[:, 1], 'o-', label='validation')

plt.legend()
plt.title('final accuracy vs batch size')
plt.xlabel('batch size')
plt.ylabel('final acc')
#plt.ylim([0.4, 1])
plt.grid(visible=True)
plt.savefig(path + 'Final Accuracy vs Batch Size.png')
#plt.savefig(path + 'Final Accuracy vs Batch Size (adjusted).png')