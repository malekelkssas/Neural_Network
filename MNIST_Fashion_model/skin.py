from  __future__  import absolute_import,division,print_function,unicode_literals

import os

#import pandas as pd
from IPython.display import clear_output
from keras.utils import to_categorical
from six.moves import urllib
#import tensorflow_probability as tfp
from keras import datasets,layers,models


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing as pre

def fc():
    return tf.compat.v2.feature_column


# organizing data ------------------

directory_path = r"D:\Github\Neural_Network_models\Train"

# Define the image size and batch size for training
img_size = (256, 256)
batch_size = 32
# directory_path=os.path.normpath(directory_path)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory_path,
    # labels='inferred',
    # label_mode='int',
    # color_mode='rgb',
    batch_size=batch_size
    # image_size=img_size,
    # shuffle=True,
    # seed=42,
    # validation_split=0.2,
    # subset='training'
)

# Assign the dataset to the X variable
images = []
labels = []
# Iterate over the batches in the dataset
for batch in train_dataset:
    # Extract the images from the batch and append to the list
    batch_images = batch[0].numpy()
    labels.append(batch[1].numpy())
    images.append(batch_images)



images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)



train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    test_size=0.2  # Set the proportion of data to use for testing
    # random_state=42  # Set the random seed for reproducibility
)

class_names = ['real','garden']
train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# print(train_labels[0])

# building the model -----------

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',input_shape=(256,256,3)))  #model.add(amount of filters , simple size,activation function ,input_shape=(height,width,dimension))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))         #here we don't need the input shape because the model figure out from the prev layer
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

#4adding dense layer
#____________________
#after extracting features from the prev layers we add a way to classify them

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #hidden layer (2) ->128 neurons (it's preferable to be smaller than the prev layer(2*2*64)neurons , but it depends on the problem) , activation function I can choose one of the most common three
model.add(layers.Dense(2))     # 10 because we have 10 classes

model.summary()

#5-Train and evaluate
#___________________
model.compile(
    optimizer="adam",   #optimizer function
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #loss function
    metrics=['accuracy']    # what I want the output to be
)
history = model.fit(train_images,train_labels,epochs=1,validation_data=(test_images,test_labels))   #Train

test_loss , test_accuracy = model.evaluate(test_images,test_labels,verbose=2)
print("test accuracy = ",test_accuracy*100,"%")


#6- prediction
#_____________

predictions = model.predict(test_images)

while True:
    i = input("choose an integer number: ")
    if i.isdigit():
        i = int(i)
        tmp = np.argmax(predictions[i])  # this will give me the index of the highest probability
        print("the model prediction is :",class_names[tmp])
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.show()
        # print(test_labels[i])
        plt.xlabel(class_names[test_labels[i]])

    else:
        break