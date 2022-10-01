
# using pre-trained models technique
# ______________________________________
# the idea is to use convolutional neural network that have trained already before

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras
tfds.disable_progress_bar()

# 1-organizing data
# __________________

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    "cats_vs_dogs",
    split=['train[:80%]', "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)

get_label_name = metadata.features['label'].int2str  # create a function object that we can use to get labels
for image, label in raw_train.take(2):  # taking to image from dataset and display
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

# 2-resizing images
# __________________
# It is much better to make the images smaller than bigger, even if I will loss some details
img_size = 160  # all images will be resized to 160*160


def format_example(image, label):  # return image that is reshaped to img_size
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # 127.5 is the half of 255
    image = tf.image.resize(image, (img_size,img_size))
    return image, label


# 3-now applying format_example function for all images
# ____________________________________________________

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):  # taking to image from dataset and display
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

#4-shuffle and batch the images
#______________________________
batch_size = 32
shuffle_buffer_size = 1000
train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

# 5-picking pretrained model
# __________________________
# the model we are going to use as the convolutional base for our model is the (MobileNet V2). This model is trained
# on 1.4 million images and has 1000 different classes
# I want to use only the convolutional base of this model
# so when I load the model I don't want to download the classification layers

img_shape = (img_size,img_size,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()  # the output of this network (Output Shape)=  (None, 5, 5, 1280)
for image,_ in train_batches.take(1):        # at this point the output shape should be (32, 5, 5, 1280)
    pass

    print(base_model(image).shape)



# 6-freezing the base
# ___________________
base_model.trainable = False

# 7-adding our classifier
# _________________________
global_average_layers = keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)  # because we want the model to choose one of the two classes (cats,dogs)
model = keras.Sequential([
    base_model,
    global_average_layers,
    prediction_layer]
)


# # 8-train the model and test
# # it takes long time to train this model, so after I trained this model you can use (new_model) variable and comment this part(comment until line 115)
# # ________________
base_learning_rate = 0.0001  # it is very low because I don't want to make major change in biases weight for the original model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # I use this binary function because I am choosing from two different classes
    metrics=["accuracy"]
)
model.summary()

history = model.fit(train_batches, epochs=4, validation_data=validation_batches)  # train the model

print("the accuracy : ", history.history["accuracy"] * 100, '%')

model.save("dogs_vs_cats.h5")
new_model = keras.models.load_model("dogs_vs_cats.h5")

loss, accuracy = new_model.evaluate(test_batches)
print('Test accuracy :', accuracy*100,'%')

# 8- prediction
# _____________


image_batch, label_batch = test_batches.as_numpy_iterator().next()
predictions = new_model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)
while True:
    i = input("choose an integer number: ")
    if i.isdigit():
        i = int(i)
        print("the model prediction is ", get_label_name(predictions[i]))
        plt.figure()
        plt.imshow(image_batch[i])
        plt.title(get_label_name(label_batch[i]))
        plt.show()

    else:
        break
