from  __future__  import absolute_import,division,print_function,unicode_literals


#import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
#import tensorflow_probability as tfp


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def fc():
    return tf.compat.v2.feature_column

#My_first_neural_network

#here we will use MNIST Fashion dataset -> 60000 images for training and 10000 images for testing(closthing articles)
#it's built in keras

#1-organizing data
#__________________

fashion_mnist = keras.datasets.fashion_mnist    #load dataset
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()   #split into testing and training

#print(train_images.shape)       #output = (60000, 28, 28) -> that means 60000 images that made up 28*28 pixels (784 in total)
#print(train_images[0,23,23])    #output = 194 -> (this is one pixel)the output must br between 0 (black) and 255(white) it likes RGP value
# print(train_labels[:10])        #this is the first 10 training labels ->it's ranging between 0 and 9 each integer represents a specific article of clothing
class_names =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#this is how the images look like
#__________________________________
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# verify that the data is in the correct format
# display the first 25 images from the training set and display the class name below each image.
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
#____________________

#2-Data preprocessing
#____________________
# it's important step in neural network ,
# we always try to squeeze our values to small numbers as it's easier for our model to process->
# here we will squeeze our values between [0,1] ->
#this must happen for the training and testing data

train_images = train_images/255.0
test_images = test_images/255.0

#3-building the model
#___________________

model = keras.Sequential(
    # here we will define the layers
    [
        keras.layers.Flatten(input_shape=(28,28)),  #input layers (1)
        keras.layers.Dense(128,activation='relu'),  #hidden layer (2) ->128 neurons (it's preferable to be smaller than the input layer(784)neurons but it depends on the problem) , activation function i can choose one of the most common three
        keras.layers.Dense(10,activation='softmax')  # output layer (3)-> 10 because we have 10 classes
    ])

#4-compile the model
#___________________
model.compile(
    optimizer='adam',    #optimizer function
    loss="sparse_categorical_crossentropy",     #loss function
    metrics=['accuracy']                        # what I want the output to be
)

#5-training and testing the model
#____________________
model.fit(train_images,train_labels,epochs=10)  #training

test_loss , test_accuracy = model.evaluate(test_images,test_labels,verbose=1)
print("Test accuracy :", test_accuracy*100,'%')


#6- prediction
#_____________

predictions = model.predict(test_images)

tmp = np.argmax(predictions[0])    #this will give me the index of the highest probability
print(class_names[tmp])

plt.figure()
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.xlabel(class_names[test_labels[0]])
plt.show()
