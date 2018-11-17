
# coding: utf-8

# ## Imports

# In[4]:


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


# ## Helper functions
# Some functions to make life a bit easier.

# In[5]:


# Function for displaying progress
def display_progress(total, current):
    os.system('clear')
    percent = str((current / total) * 100)
    print(percent + "%")


# In[6]:


# Plotting function for plotting images
def plot_images(images, titles):
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


# ## Reading the image data
# Handy tutorial for pydicom:
# https://mscipio.github.io/post/read_dicom_files_in_python/

# In[ ]:


# Get all the files from the data/stage_1_train_images folder
train_dcm_file_names = os.listdir('data/stage_1_train_images')
# Put all the files in one array
train_dcm_files = []
for i in range(0, len(train_dcm_file_names)):
    # Add each file to the array and display the progress
    # This takes a while
    name = train_dcm_file_names[i]
    file = pdcm.read_file('data/stage_1_train_images/' + name)
    train_dcm_files.append(file)
    display_progress(len(train_dcm_file_names), i)


# ## Quick visual of some scans

# In[ ]:


images = []
titles = []
for i in range(0, 4):
    # Get image data
    image = train_dcm_files[i].pixel_array
    title = train_dcm_files[i].PatientID
    images.append(image)
    titles.append(title)

plot_images(images, titles)


# ## Prepare the data
# Let's load the labels and prepare the data.

# In[2]:


labels = pd.read_csv('data/stage_1_train_labels.csv', header=None)

# Function for finding item in array (returns index)
def find_item(array, item):
    for i in range(0, len(array)):
        if item == array[i][0]:
            return i

# Align the label array with the dcm files and show progress
# This takes a while
train_labels = []
for i in range(0, len(train_dcm_files)):
    file = train_dcm_files[i]
    patient_id = file.PatientID
    index = find_item(labels.values, patient_id)
    train_labels.append(labels.values[index])
    display_progress(len(train_dcm_files), i)


# In[1]:


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


# In[ ]:


# Reshape the images and normalize
# train_images = [x.pixel_array / 255.0 for x in train_dcm_files]
train_images = []
for i in range(0, len(train_dcm_files)):
    train_images.append(train_dcm_files[i].pixel_array / 255.0)
    display_progress(len(train_dcm_files), i)


# ## Build the model
# Using keras

# In[260]:


def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(200, activation=tf.nn.relu))
    model.add(keras.layers.Dense(200, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))
    return model

model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, final_train_labels)
