import cv2
import os
from PIL  import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


# Initialise the path of the dataset
image_directory = 'datasets/'

#Creating a list inside a new folder
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []

INPUT_SIZE=64

#print(no_tumor_images)


#path = 'no0.jpg' 
#print(path.split('.')[1]) #here it goes from 0 to 1 and check if the image is jpg or not 

#Now We are going to iterate through all the images of no folder
for i, image_name in enumerate(no_tumor_images):
     #Check if image is jpg or not
    if(image_name.split('.')[1] == 'jpg'):
        #for reading the images we use open cv2
        image = cv2.imread(image_directory + 'no/' + image_name)
        #to read the image we are going to convert it in array format
        image = Image.fromarray(image, 'RGB')
        #now Resizing the images because each images may have different size
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        #Each image in dataset will appended into numpy array
        dataset.append(np.array(image))
         #giving them a label as 0 means NO They do not have tumor
        label.append(0)

#Now We are going to iterate through all the images of yes folder
for i, image_name in enumerate(yes_tumor_images):
    #Check if image is jpg or not
    if(image_name.split('.')[1] == 'jpg'):
        #for reading the images we use open cv2
        image = cv2.imread(image_directory + 'yes/' + image_name)
        #to read the image we are going to convert it in array format
        image = Image.fromarray(image, 'RGB')
        #now Resizing the images because each images may have different size
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        #Each image in dataset will appended into numpy array
        dataset.append(np.array(image))
        #giving them a label as 1 means YES They have tumor
        label.append(1)

#print(len(dataset)) #prints the length if not len then dataset
#print(len(label))

#converting datasets and label into numpy array 
dataset = np.array(dataset)
label = np.array(label)

#Now we actually need to divide the dataset into train test and split
#so for that import sklearn scikit library -- train_test_split

#dividing dataset into 80% and 20%
#80% training and 20% testing

#parameters= array - dataset,array - label, test_size = 20% data for the testing
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 0)

#print(x_train.shape) #number of train set - x 80% images -->2400
# (2400 -- images, 64 - dimension, 64 - dimension, 3 - RGB - scale)
#Reshape = (n, image_width, image_height, n_channels)
#print(y_train.shape) #number of train set - y

#print(x_test.shape) # --> 20% of images 600
#print(y_test.shape) 

#normalise the data for training purposes
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1) #this is one of binary class problem

#categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#MODEL BUILDING

#1 Sequential Model
#64 64 3 --> our images
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = (INPUT_SIZE,INPUT_SIZE, 3)))
#32 - filter , kernal 3, 3, inputsize - 64,  64 , 3 - RGB
model.add(Activation('relu'))
#for activation we are using relu function
model.add(MaxPooling2D(pool_size=(2,2)))


#random model for training our program

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten to make all the images into one factor
model.add(Flatten())
#adding the dense layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) #--> 1 because our answer is yes or no thats why not 2 
#2 nahi de skte kyuki 1 hi hoga or categorical format m nhi jayega
#and because we are using binary classification problem

#model.add(Activation('sigmoid'))
model.add(Activation('softmax')) #--> as the last layer

#if we use binary crossEntropy
#Binary CrossEntropy = 1, sigmoid
#Categorical Cross entropy = 2, softmax -->because of yes or no types of prediction


#compile the model it will take loss as binary cross entropy
#optimiser is adam popular
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #-->byclassification

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10,validation_data=(x_test, y_test), shuffle = False)
#batchsize --> number of batch - 16
#verbose --> will show you an animated progress bar like this: =======================
#epochs --> indicates the number of passes of the entire training dataset the machine learning algorithm has completed.
#shuffle --> in order to create more representative training and testing sets

#model.save('BrainTumor10Epochs.h5') #by binary lassification 

#now by category
#save the model
model.save('BrainTumor10EpochsCategorical.h5')  # by categorical classification