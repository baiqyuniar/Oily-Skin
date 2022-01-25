from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import shutil
import os
from PIL import Image
import json

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

port = int(os.environ.get('PORT', 5000))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.get_session
cv2.ocl.setUseOpenCL(False)

label = ["Normal Face","Oily Face"]

def detect_face(img):
    str = ""
    face_img = img.copy()
    #Reads face using haarcascades 
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt2.xml')
    face_rects = face_cascade.detectMultiScale(face_img,
                                               scaleFactor=1.2,
                                               minNeighbors=5,
                                               minSize=(150, 150)
                                              )

# to add border to photo
    borderType = cv2.BORDER_CONSTANT
    if len(face_rects)==0:
        global faces
        faces = face_img
        #Size for border
        top = int(0.05*faces.shape[0])
        bottom = top
        left = int(0.05*faces.shape[1])
        right=left
        faces=cv2.copyMakeBorder(faces,top,bottom,left,right,borderType,None,[0,0,0])
        str = "Wajah tidak terdeteksi"
    else: 
        for (x,y,w,h) in face_rects: 
            cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,0,0), 10)
            faces = face_img[y:y + h, x:x + w]
            #Size for border
            top = int(0.05*faces.shape[0])
            bottom = top
            left = int(0.05*faces.shape[1])
            right=left
            faces=cv2.copyMakeBorder(faces,top,bottom,left,right,borderType,None,[0,0,0])
        str = "Wajah terdeteksi"
    return faces, str

  # Function to predict the class of the photo taken
def return_prediction(model,detect_face,file):
    # Read the input image
    test = cv2.imread(file)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    result, face = detect_face(test)
    result = cv2.resize(result, (150, 150))
    result = np.asarray(result)
    result = np.expand_dims(result, axis=0)
    result = result/255
    prediction_prob = model.predict(result)
    # prediction_prob=np.round(prediction_prob)
    # Output prediction
    str = label[int(np.round(prediction_prob))]
    val = prediction_prob[0][0]
    return str, val, face

model = load_model('model/model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, template_folder='template')
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def main():
    folder = 'static/upload/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return render_template('main/index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        if not allowed_file(file.filename):
            flash('Unsupported file')
            return redirect('/')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fileloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fileloc)
            str, val, face = return_prediction(model,detect_face,fileloc)
            return render_template('predict/index.html', detect = face, result = str, pred = val, file = fileloc)

#For API usage
@app.route('/api', methods=['POST'])
def api():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fileloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fileloc)
            str, val, face = return_prediction(model,detect_face,fileloc)
            folder = 'static/upload/'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            return jsonify({'msg': 'success', 'face' : face, 'result' : str, 'value' : json.dumps(val.item())})

if __name__ == '__main__':
    app.run(debug=True)


