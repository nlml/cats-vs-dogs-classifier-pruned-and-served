from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import shutil
from pytorch_model import Model
from time import time
import json
from io import BytesIO
from PIL import Image


# Make folder for storing user-uploaded images
if not os.path.exists('/app/static'):
    os.makedirs('/app/static')

# Instantiate the Pytorch model
model = Model()

app = Flask(__name__)

@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        f.save(fname)
        res = model.predict_image_from_path(fname)
        ext = fname.split('.')[-1]
        for i in os.listdir('/app/static'):
            if i.startswith('uploaded_img'):
                os.remove(os.path.join('/app/static', i))
        uploaded_img_fname = f'uploaded_img.{ext}'
        shutil.copyfile(fname, f'/app/static/{uploaded_img_fname}')
        os.remove(fname)
        t = int(time())
        return '''
            <!DOCTYPE html>
            <html>
                <head>
                    <style>
                        img
                        {
                            max-width:450px;
                            max-height:450px;
                        }
                        html *
                        {
                           font-size: 1em !important;
                           color: #000 !important;
                           font-family: Helvetica !important;
                        }
                    </style>
                </head>
            <body>
            BODYBODYBODY
            <a href="/upload">Upload a different image...</a>
            </body>
            </html>
            '''.replace('BODYBODYBODY', f'<img src="/static/{uploaded_img_fname}"></img>{res}')

@app.route('/classify', methods=['POST'])
def api_message():

    if request.headers['Content-Type'] == 'text/plain':
        return "Text Message: " + request.data

    elif request.headers['Content-Type'] == 'application/json':
        result, ms = model.predict_image_from_tensor(request.json['tensor'])
        return json.dumps({'predictions': result, 'classes': ['Cat', 'Dog']})

    elif request.headers['Content-Type'] == 'application/octet-stream':
        file_data = BytesIO(request.data)
        img_pil = Image.open(file_data)
        result, ms = model.predict_image_from_pil(img_pil)
        return json.dumps({'predictions': result, 'classes': ['Cat', 'Dog']})

    else:
        return "415 Unsupported Media Type"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
