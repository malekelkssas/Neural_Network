from  __future__  import absolute_import,division,print_function,unicode_literals


#import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
#import tensorflow_probability as tfp


import tensorflow as tf
from tensorflow import keras
from keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt

def fc():
    return tf.compat.v2.feature_column

#1-organizing data
#____________________
(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()   #split into testing and training
train_images , test_images = train_images/255.0 , test_images/255.0 #normalize data between 0 and 1
class_names = ['Airplane',"Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

#2- making CNN Architecture
#____________________
#to make Architecture we stack convolutional layers and punch of max pooling layers together (after each convolutional layer a max pooling layer)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',input_shape=(32,32,3)))  #model.add(amount of filters , simple size,activation function ,input_shape=(height,width,dimension))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))         #here we don't need the input shape because the model figure out from the prev layer
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
#model.add(layers.MaxPooling2D(2,2))
#pading happen in these layers
model.summary()     #look at our model so far

#4-adding dense layer
#____________________
#after extracting features from the prev layers we add a way to classify them

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #hidden layer (2) ->128 neurons (it's preferable to be smaller than the prev layer(2*2*64)neurons , but it depends on the problem) , activation function I can choose one of the most common three
model.add(layers.Dense(10))     # 10 because we have 10 classes

model.summary()

#5-train and evaluate
#___________________
model.compile(
    optimizer="adam",   #optimizer function
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #loss function
    metrics=['accuracy']    # what I want the output to be
)
history = model.fit(train_images,train_labels,epochs=4,validation_data=(test_images,test_labels))   #train

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
        print("the model prediction is ",class_names[tmp])
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[i][0]])
        plt.show()
    else:
        break
