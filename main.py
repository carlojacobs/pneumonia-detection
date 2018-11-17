# Carlo Jacobs
# Sat 17 Nov 2018

# Imports
import numpy as np # for doing math stuff
import tensorflow as tf # for building the modle
from tensorflow import keras # for building the model
import pandas as pd # for reading csv
import csv # for writing csv
import pydicom as pdcm # for reading the image data
import matplotlib.pyplot as plt # for visualizing the data
import os # for file manipulation
from IPython.display import clear_output, display # for clearing output and displaying
import json # saving variables because I'm lazy

# Helper functions
# Note: these function may not be fully objective when it comes to their input, they may expect a certain shape of data
def display_progress(total, current):
    """Displays the progress"""
    os.system('clear')
    percent = str((current / total) * 100)
    print(percent + "%")

def find_item_index(array, item):
    """Finds the index of an item in an array"""
    for i in range(0, len(array)):
        if item == array[i][0]:
            return i

def find_item_in_array(array, item):
    """Returns if an item is in an array"""
    for i in range(0, len(array)):
        if item == array[i][0]:
            return true
    return false

def plot_images(images, titles):
    """Plost a set of images"""
    fig_shape = int(np.ceil(len(images) / 2))
    fig = plt.figure(figsize=(20, 20))
    for i in range(0, len(images)):
        img = images[i]
        ax = fig.add_subplot(fig_shape, fig_shape, i + 1)
        if len(titles) != 0:
            plt.title(titles[i])
        ax.imshow(img, cmap=plt.cm.binary)
        ax.axis('off')
    plt.show()

def read_dcm_files(names):
    """Takes an array of dcm file names and reads the dcm files"""
    print("[*] Reading the dcm files...")
    read = lambda name : pdcm.read_file('data/stage_1_train_images/' + name)
    files = [read(x) for x in names]
    return files

def write_array_to_json(array, filename):
    """Takes a 2d array and writes it to a json file"""
    json_dict = {}
    for i in range(0, len(array)):
        item = array[i]
        sub_dict = {
            "x": array[1],
            "y": array[2],
            "width": array[3],
            "height": array[4],
            "target": array[5]
        }
        json_dict[str(item[0])] = sub_dict

    with open(filename, 'w') as outfile:
        json.dump(json_dict, outfile)

def read_json_to_array(filename):
    """Reads a json file and puts the data in an array"""
    with open(filename) as json_file:
        array = []
        data = json.load(json_file)
        for key, value in data:
            array.append(value)
        return array

def format_data(files, labels):
    """Takes an array of dcm files and an array of the labels and properly manages the data"""
    print("[*] Formatting the data...")
    print("[*] Aligning the labels")
    # Align the labels with the dcm files
    for i in range(0, len(files)):
        file = files[i]
        patient_id = file.PatientID
        index = find_item_index(labels.values, patiend_id)
        labels[i] = labels.values[index]

    print("[*] Normalizing the pixeldata")
    # Normalize the images
    files = files.pixel_array / 255.0

    # Return the data
    return files, labels

# Load and format the data
train_dcm_file_names = os.listdir('data/stage_1_train_images')
test_dcm_file_names = os.listdir('data/stage_1_test_images')

train_dcm_files = read_dcm_files(train_dcm_file_names)
# test_dcm_files = read_dcm_files(test_dcm_file_names)


# Get the labels
train_labels = pd.read_csv('data/stage_1_train_labels.csv', header=None)

print(train_labels.shape)
print(train_labels[0])
print(train_labels[0][0])

write_array_to_json(train_labels, 'train_labels.json')
