import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "../logs/experiment1_local_iter_types/uniform/"
final_acc = pd.read_csv(path + "final_acc.csv")
final_loss = pd.read_csv(path + "final_loss.csv")
#beta_samples = np.logspace(start=np.log10(0.05), stop=1, num=15, base=10)   #from 0.05 to 10
#beta_samples = np.logspace(start=-3, stop=np.log10(2), num=15, base=10)  # from 0.001 to 2
beta_samples = np.logspace(start=-1, stop=np.log10(2), num=15, base=10)  # from 0.1 to 2


plt.figure(1, figsize=(10, 10))
plt.plot(beta_samples, final_loss.iloc[:-1, 2], 'o-', label='train (dirichlet)')
plt.plot(beta_samples, final_loss.iloc[:-1, 3], 'o-', label='train after updating (dirichlet)')
plt.plot(beta_samples, final_loss.iloc[:-1, 1], 'o-', label='test (dirichlet)')
plt.plot(beta_samples, [final_loss.iloc[-1, 2]]*len(final_loss.iloc[:-1, 1]), '--', label='train (uniform)')
plt.plot(beta_samples, [final_loss.iloc[-1, 3]]*len(final_loss.iloc[:-1, 1]), '--', label='train after updating (uniform)')
plt.plot(beta_samples, [final_loss.iloc[-1, 1]]*len(final_loss.iloc[:-1, 1]), '--', label='test (uniform)')
plt.legend()
plt.title('final Loss vs beta')
plt.xlabel('beta')
plt.ylabel('final Loss')
plt.grid(visible=True)
plt.savefig(path + 'Final Loss vs Beta.png')

plt.figure(2, figsize=(10, 10))
plt.plot(beta_samples, final_acc.iloc[:-1, 2], 'o-', label='train (dirichlet)')
plt.plot(beta_samples, final_acc.iloc[:-1, 3], 'o-', label='train after updating (dirichlet)')
plt.plot(beta_samples, final_acc.iloc[:-1, 1], 'o-', label='validation (dirichlet)')
plt.plot(beta_samples, [final_acc.iloc[-1, 2]]*len(final_loss.iloc[:-1, 1]), '--', label='train (uniform)')
plt.plot(beta_samples, [final_acc.iloc[-1, 3]]*len(final_loss.iloc[:-1, 1]), '--', label='train after updating (uniform)')
plt.plot(beta_samples, [final_acc.iloc[-1, 1]]*len(final_loss.iloc[:-1, 1]), '--', label='validation (uniform)')
plt.legend()
plt.title('final accuracy vs beta')
plt.xlabel('beta')
plt.ylabel('final acc')
plt.grid(visible=True)
plt.savefig(path + 'Final Accuracy vs Beta.png')

plt.figure(3, figsize=(10, 10))
plt.loglog(beta_samples, final_loss.iloc[:-1, 2], 'o-', label='train (dirichlet)')
plt.loglog(beta_samples, final_loss.iloc[:-1, 3], 'o-', label='train after updating (dirichlet)')
plt.loglog(beta_samples, final_loss.iloc[:-1, 1], 'o-', label='validation (dirichlet)')
plt.loglog(beta_samples, [final_loss.iloc[-1, 2]]*len(final_loss.iloc[:-1, 1]), '--', label='train (uniform)')
plt.loglog(beta_samples, [final_loss.iloc[-1, 3]]*len(final_loss.iloc[:-1, 1]), '--', label='train after updating (uniform)')
plt.loglog(beta_samples, [final_loss.iloc[-1, 1]]*len(final_loss.iloc[:-1, 1]), '--', label='validation (uniform)')
plt.legend()
plt.title('final Loss vs beta')
plt.xlabel('beta (log scale)')
plt.ylabel('final Loss (log scale)')
plt.grid(visible=True)
plt.savefig(path + 'Final Loss vs Beta (log scale).png')

plt.figure(4, figsize=(10, 10))
plt.semilogx(beta_samples, final_acc.iloc[:-1, 2], 'o-', label='train (dirichlet)')
plt.semilogx(beta_samples, final_acc.iloc[:-1, 3], 'o-', label='train after updating (dirichlet)')
plt.semilogx(beta_samples, final_acc.iloc[:-1, 1], 'o-', label='validation (dirichlet)')
plt.semilogx(beta_samples, [final_acc.iloc[-1, 2]]*len(final_loss.iloc[:-1, 1]), '--', label='train (uniform)')
plt.semilogx(beta_samples, [final_acc.iloc[-1, 3]]*len(final_loss.iloc[:-1, 1]), '--', label='train after updating (uniform)')
plt.semilogx(beta_samples, [final_acc.iloc[-1, 1]]*len(final_loss.iloc[:-1, 1]), '--', label='validation (uniform)')
plt.legend()
plt.title('final accuracy vs beta')
plt.xlabel('beta (log scale)')
plt.ylabel('final acc')
plt.grid(visible=True)
plt.savefig(path + 'Final Accuracy vs Beta (log scale).png')
