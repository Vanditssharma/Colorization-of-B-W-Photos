import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import base64

app = Flask(__name__)
prototxt = "model/colorization_deploy_v2.prototxt"
model = "model/colorization_release_v2.caffemodel"
points = "model/pts_in_hull.npy"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    imagefile= request.files['imagefile'] 
    image_path = "images/" + imagefile.filename
    imagefile.save(image_path)

    testpath = 'images'
    files = os.listdir(testpath)
    for idx, file in enumerate(files):
        
        image = cv2.imread(testpath+'/'+file)
        os.remove(image_path)
        save = image
        window_name = 'image'
    
        cv2.imshow(window_name, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        retval, buffer = cv2.imencode('.jpg', colorized)
        jpg_as_text = base64.b64encode(buffer)

        return jpg_as_text
        # return render_template('index.html', prediction_text=jpg_as_text)

if __name__ == "__main__":
    app.run(debug=True)
