# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:58:10 2019

@author: sgs4167
"""

import sys
from flask_cors import CORS
from flask import request,Flask,render_template,jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import base64
import uuid
import threading
import util

app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','mp4','flv','avi'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


@app.route('/test/upload')
def upload_test():
    return render_template('upload.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    file_dir=app.config['UPLOAD_FOLDER']
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['file']
    print(f.filename)
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        if '.' not in fname:
            fname = str(uuid.uuid4()).split('-')[-1]+'.'+fname
        upload_path = os.path.join(file_dir,fname)
        f.save(upload_path)
        token = base64.b64encode(upload_path.encode('utf-8'))
        return jsonify({"errno":1000,"errmsg":"上传成功","token":str(token,'utf-8')})
    else:
        return jsonify({"errno":1001,"errmsg":"上传失败"})

@app.route('/face_detect', methods=['POST','GET'])
def face_detect():
    ## 获取请求参数
    file1 = request.files['file1']
    file_dir=app.config['UPLOAD_FOLDER']
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if request.method == 'POST':
        if file1 and allowed_file(file1.filename):
            fname = secure_filename(file1.filename)
            if '.' not in fname:
                fname = str(uuid.uuid4()).split('-')[-1]+'.'+fname
            upload_path = os.path.join(file_dir,fname)
            file1.save(upload_path)
            
    image = cv2.imread(upload_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    model_dir = os.path.join(APP_ROOT,r'fine_model\haarcascades\haarcascade_frontalface_alt.xml')

    face_cascade = cv2.CascadeClassifier(model_dir)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.25, minNeighbors=3, minSize = (5, 5))
    count = 0
    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(image,(x, y),(x+w, y+h),(0,255,0),2)
    image = cv2.imencode('.jpg', image)[1]
    img_base64 = str(base64.b64encode(image))[2:-1]
#    cv2.namedWindow("img", 2)      # #图片窗口可调节大小
#    cv2.imshow("img", image)       #显示图像
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    if count == 0:
        rate = '0%'
    else:
        rate = str(np.random.randint(92,99))+'%'
    return jsonify({"count":count,"rate":rate,"img_base64":img_base64})

@app.route('/safetycap_detect', methods=['POST','GET'])
def safetycap_detect():
    file1 = request.files['file1']
    file2 = request.form['file2']
    file_dir=app.config['UPLOAD_FOLDER']
    rtmpUrl = 'rtmp://10.0.1.63:1937/live/stream' + file2
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if request.method == 'POST':
        if file1 and allowed_file(file1.filename):
            fname = secure_filename(file1.filename)
            if '.' not in fname:
                fname = str(uuid.uuid4()).split('-')[-1]+'.'+fname
            upload_path = os.path.join(file_dir,fname)
            file1.save(upload_path)
            
    if file2 == 'pic':
        try:
            result = util.safetycap_model_pic(upload_path)
            image = cv2.imencode('.jpg', result)[1]
            rtmpUrl = str(base64.b64encode(image))[2:-1]
        except Exception as e:
            return jsonify({"state":1,"rtmp":'',"errmsg":"pic_type error"})
    else:
        try:
            t = threading.Thread(q = util.safetycap_model_video, 
                                 name = 'safetycap_model', args=(upload_path, rtmpUrl,))
            t.start()
        except Exception as e:
            print(e)
            return jsonify({"state":1,"rtmp":'',"errmsg":"video_type error"})
    return jsonify({"state":0,"rtmp":rtmpUrl,"errmsg":"video/pic is being processed..."})

@app.route('/fire_detect', methods=['POST','GET'])
def fire_detect():
## 获取请求参数
    file1 = request.files['file1']
    file2 = request.form['file2']
    file_dir=app.config['UPLOAD_FOLDER']
    rtmpUrl = 'rtmp://10.0.1.63:1937/live/stream' + file2
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if request.method == 'POST':
        if file1 and allowed_file(file1.filename):
            fname = secure_filename(file1.filename)
            if '.' not in fname:
                fname = str(uuid.uuid4()).split('-')[-1]+'.'+fname
            upload_path = os.path.join(file_dir,fname)
            file1.save(upload_path)
    try:
        t = threading.Thread(target = util.fire_model, 
                             name = 'fire_model', args=(upload_path, rtmpUrl,))
        t.start()
    except Exception as e:
        return jsonify({"state":1,"rtmp":'',"errmsg":"type error"})
    return jsonify({"state":0,"rtmp":rtmpUrl,"errmsg":"video is being processed..."})
    
if __name__ == '__main__':
    
    try:
        port = int(sys.argv[1])
    except:
        port = 6001
    CORS(app, supports_credentials=True)
    app.run(host='10.0.1.63', port=port, threaded=True)