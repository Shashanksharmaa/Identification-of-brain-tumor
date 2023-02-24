import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
#render template is nothing but for rendering the template file 
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/ ')

def get_className(classNo):
    if classNo == 0:
        return "NO BRAIN TUMOR"
    elif classNo == 1:
        return "BRAIN TUMOR DETECTED"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis = 0)
    result = (model.predict(input_img) > 0.5).astype("int32")
    return result

@app.route('/', methods = ['GET']) #get is used for insecure one browse the images
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST']) #uploads the images
#get is insecure
#post is secure
def upload():
    if request.method == 'POST': #using request as imported using flask framework
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join( #join the prediction folder
            basepath, 'uploads', secure_filename(f.filename)) #uploads everything on uploads file
        #save file name as file path
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug = True)

