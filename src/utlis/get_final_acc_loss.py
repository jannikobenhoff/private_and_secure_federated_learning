import os
import pandas as pd

# use read_logs.py first
root_path = '../logs/experiment2_same_local_iteration/50clients/'
# change the folder path to the folder which contains generated log folders only
folder_names = os.listdir(root_path)
subfolder_names = ['loss', 'acc']
csv_files_names = os.listdir(root_path+folder_names[0]+'/'+subfolder_names[0])
#csv_files_names = csv_files_names[:-1]
for i in range(2):
    final_values = {}
    for folder_name in folder_names:
        final_values_folder = []
        for file in csv_files_names:
            file_path = root_path + folder_name + '/' + subfolder_names[i] + '/' + file
            df = pd.read_csv(file_path)
            final_value = df.iloc[-1, -1]
            # extract values from specific epoch
            """
            if file == 'train_clients_average_federator.csv':
                final_value = df.iloc[54, -1]  # starts with step 0, others start with step 1
            else:
                final_value = df.iloc[53, -1]
            """
            final_values_folder.append(final_value)
        final_values[folder_name] = final_values_folder
    #print(final_values)
    final_values_df = pd.DataFrame(final_values)
    final_values_df = final_values_df.T
    final_values_df.index = folder_names
    final_values_df.columns = csv_files_names
    final_values_df.to_csv(root_path+'final_'+subfolder_names[i]+'.csv')
    #final_values_df.to_csv(root_path + 'final_' + subfolder_names[i] + '_1.csv')

