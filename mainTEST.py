import cv2 #to actually load image
from keras.models import load_model 
from PIL import Image
import numpy as np

#load the model from the directory
model = load_model('BrainTumor10EpochsCategorical.h5')

#read the image
image = cv2.imread('/Users/shashanksharma/Desktop/Brain Tumor WebApp/pred/pred0.jpg')

#convert above image into array because we have trained an array then resize then append to numpy array

img = Image.fromarray(image)

#resize each image
img = img.resize((64, 64))

#convert each image in numpy array
img = np.array(img)

#print(img) #predicting tumor based on np array
#but now on model
#expand the dimension and predict!

input_img = np.expand_dims(img, axis = 0)

#result = (model.predict(input_img) > 0.5).astype("int32") #-->Binary classification

result = np.argmax(model.predict(input_img), axis=-1) #-->Categorical classification

print(result)


