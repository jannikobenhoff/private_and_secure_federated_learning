import os
from tensorboard.backend.event_processing import event_accumulator
import csv
import tensorflow as tf


def extract_data_and_save_csv(path, tag, output_dir, output_filename):
    files = os.listdir(path)
    for file in files:
        if file.startswith('events.out.'):
            event_file = os.path.join(path, file)
            ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.TENSORS: 0})
            ea.Reload()
            events = ea.Tensors(tag)
            csv_filename = os.path.join(output_dir, output_filename)
            with open(csv_filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['wall time', 'step', 'value'])
                for event in events:
                    csv_writer.writerow([event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item()])


root_path = '../logs/experiment4_batch_size/rs25/'
# change the folder path to the folder which contains generated log folders only
folder_names =os.listdir(root_path)
for folder in folder_names:
    logs_dir = os.path.join(root_path, folder)
    tags = ['loss', 'accuracy']
    output_folder_names = ['loss', 'acc']
    logs_folder_names = ['test/', 'train/clients_average/', 'train/clients_average_federator/']
    output_filenames = ['test.csv', 'train_clients_average.csv', 'train_clients_average_federator.csv']
    for j in range(len(tags)):
        output_dir = os.path.join(logs_dir, output_folder_names[j])
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(logs_folder_names)):
            extract_data_and_save_csv(os.path.join(logs_dir, logs_folder_names[i]), tags[j], output_dir, output_filenames[i])

