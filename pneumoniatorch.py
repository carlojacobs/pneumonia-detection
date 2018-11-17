# Imports
import torch
import torch.optim as optim
import torch.nn as nn
from net import Net
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as pdcm # for reading the image data
import csv

# Helper functions

# Function for displaying progress
def display_progress(total, current):
    os.system('clear')
    percent = str((current / total) * 100)
    print(percent + "%")

# Function for finding item in array (returns index)
def find_item(array, item):
    for i in range(0, len(array)):
        if item == array[i][0]:
            return i

# Plotting function for plotting images
def plot_images(images, titles):
    fig_shape = int(np.ceil(len(images) / 2))
    fig = plt.figure(figsize=(5, 5))
    for i in range(0, len(images)):
        img = images[i]
        ax = fig.add_subplot(fig_shape, fig_shape, i + 1)
        if len(titles) != 0:
            plt.title(titles[i])
        ax.imshow(img, cmap=plt.cm.binary)
        ax.axis('off')
    plt.show()

# ## Reading the image data
# Handy tutorial for pydicom:
# https://mscipio.github.io/post/read_dicom_files_in_python/

def parse_train_data(batch_size):
    # Get a batch of the files from the data/stage_1_train_images folder
    train_dcm_file_names = os.listdir('data/stage_1_train_images')
    # Put all the files in one array
    train_dcm_files = []
    for i in range(0, batch_size):
        # Add each file to the array and display the progress
        # This takes a while
        name = train_dcm_file_names[i]
        file = pdcm.read_file('data/stage_1_train_images/' + name)
        train_dcm_files.append(file)
        display_progress(batch_size, i)

    labels = pd.read_csv('data/stage_1_train_labels.csv', header=None)


    # Align the label array with the dcm files and show progress
    # This takes a while
    train_labels = []
    for i in range(0, len(train_dcm_files)):
        file = train_dcm_files[i]
        patient_id = file.PatientID
        index = find_item(labels.values, patient_id)
        train_labels.append(labels.values[index])
        display_progress(len(train_dcm_files), i)

    # Reshape the data into the correct form
    # Reshape the labels
    final_train_labels = []
    for i in range(0, len(train_labels)):
        target = int(train_labels[i][5])
        if target == 0:
            final_train_labels.append([1, 0])
        else:
            final_train_labels.append([0, 1])
        display_progress(len(train_labels), i)

    # Reshape the images and normalize
    # train_images = [x.pixel_array / 255.0 for x in train_dcm_files]
    train_images = []
    for i in range(0, len(train_dcm_files)):
        train_images.append(train_dcm_files[i].pixel_array / 255.0)
        display_progress(len(train_dcm_files), i)


    return torch.Tensor(train_images), torch.Tensor(final_train_labels)

input_data, output_data = parse_train_data(300)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(input_data.shape, output_data.shape)

def train(epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(input_data)):
            inputs = input_data[i]
            outputs = net(inputs)
            label = output_data[i]
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # for i, data in enumerate(input_data, 0):
            # # get the inputs
            # inputs, labels = data


            # # zero the parameter gradients
            # optimizer.zero_grad()

            # # forward + backward + optimize
            # outputs = net(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                      # (epoch + 1, i + 1, running_loss / 2000))
                # running_loss = 0.0

    print("Finished training")

train(2)
