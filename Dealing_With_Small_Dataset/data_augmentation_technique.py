from __future__ import absolute_import, division, print_function, unicode_literals

# import pandas as pd
from IPython.display import clear_output
from six.moves import urllib
# import tensorflow_probability as tfp


import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


def fc():
    return tf.compat.v2.feature_column

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# data augmentation technique
# ______________________________
# the idea is that if I have one image I can turn this image into several images, and train pass all these images to the model
# E.G(rotate the image , flip ,stretch , compress , shift it , zoom it ...and so on)
# to know more read about ImageDataGenerator
# 1-create a data generator object that transforms images

tmp = keras.preprocessing.image.img_to_array # I use this variable because it looks like I have problem in using this function directly

(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()   #split into testing and training
train_images , test_images = train_images/255.0 , test_images/255.0 #normalize data between 0 and 1
class_names = ['Airplane',"Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

data_generator = ImageDataGenerator(    #pass some data about how i want to modify images
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

#1-pitch an image to transform
test_img = train_images[14]
img = tmp(test_img)  #convert image to numpy array
img = img.reshape((1,) + img.shape)

i=0
for batch in data_generator.flow(img , save_prefix='test',save_format='jpeg'):  #this loops run forever until we break , saving images to current directory , it uses random feature from data_generator
    plt.figure(i)
    plot = plt.imshow(tmp(batch[0]))
    i+=1
    if i > 4:   #show 4 images
        break
plt.show()
