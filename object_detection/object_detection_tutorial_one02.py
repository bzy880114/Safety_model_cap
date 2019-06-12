# -*- coding: utf-8 -*-
"""
Created on :2018/8/29 18:28

@author: sgs4176
"""
import os
import cv2
import time
#import argparse
#import multiprocessing
import numpy as np
import tensorflow as tf
import sys
import subprocess as sp
import psycopg2
import configparser
import logging
import uuid
import glob
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#from utils.app_utils import FPS, WebcamVideoStream
#from multiprocessing import Queue, Pool

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# What model to download.
#MODEL_NAME = 'frozen_inference_graph_ssd_m_v1'
#MODEL_NAME = 'frozen_inference_graph_ssd_m_v2'
MODEL_NAME = 'frozen_inference_graph_ssd_m_v1_5'
#MODEL_NAME = 'frozen_inference_graph_frcnn_inception'
#MODEL_NAME = 'ssd_mobilenet_v1_coco'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

# pgsql config
PATH_TO_PGSQL = os.path.join('data', 'pgsql.config')

# rtmp address
rtmpUrl = 'rtmp://10.0.1.63:1937/live/stream'
 
NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
category_index[1]['name']=u'无安全帽'
category_index[2]['name']=u'安全帽'

# connect to pgsql 
def connect_to_pgsql(path):
    cf = configparser.ConfigParser()
    cf.read(path)
    host = cf.get('db', 'db_host')
    port = cf.get('db', 'db_port')
    database = cf.get('db', 'db_database')
    user = cf.get('db', 'db_user')
    password = cf.get('db', 'db_password')    
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    return conn


# put result to rtmp streaming
def put_to_rtmp(rtmpUrl, sizeStr, fps_num):
    command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', sizeStr,
                '-r', str(fps_num),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                rtmpUrl]
    proc = sp.Popen(command, stdin=sp.PIPE, shell=True) #shell=
    return proc


# format detect result
def detect_result_to_pgsql(fname, fequipment_id, fwarning_image, fvideo, conn):
    cur = conn.cursor()

    nowTime = time.strftime('%Y-%m-%d %H:%M:%S')
    detect_result = 'insert into t_mon_message (fname,fwarning_time,fequipment_id,fwarning_image,fvideo) VALUES (\'{}\', \'{}\' , \'{}\', \'{}\', \'{}\')'.format(fname,nowTime, fequipment_id, fwarning_image, fvideo)
#    print('detect_result',detect_result)
    try:
        cur.execute(detect_result)
        logger.info('预警信息记录！')
    except Exception as e:
        logger.error('detect_result sql error!')
    conn.commit()
    cur.close()

# 获取电子围栏坐标
def get_fence_area(conn,equipment_id,status):
    get_area = 'select fequipment_id,fpoint,fstatus from t_mon_fence where fequipment_id=\'{}\' and fstatus=\'{}\''.format(equipment_id,status)
    cur = conn.cursor()
    try:
        cur.execute(get_area)
        logger.info('获取电子围栏最新信息！')
    except Exception as e:
        logger.error('get_fence_area sql error!')
    row = cur.fetchall()
    cur.close()
    
    return row

#更新最新截图
def update_fence_area(conn,frame,fid):
    cur = conn.cursor()
    selecr_area = 'select 1 from t_mon_equipment where fid=\'{}\''.format(fid)
    cur.execute(selecr_area)
    result = cur.fetchall()
    if result is not None:
        update_area = 'update t_mon_equipment set fimage=\'{}\' where fid=\'{}\''.format(frame,fid)
        try:
            cur.execute(update_area)
            logger.info('更新摄像头最新图片！')
        except Exception as e:
            logger.error('update_area sql error!')
    else:
        pass
    conn.commit()
    cur.close()
    
#时间格式化
def get_format_time():
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    return time_now

#删除指定图片
def del_pic(pic_dir,ext='',ignore = ''):
    if ext == '' :
        pass
    else:
        pic_list = glob.glob(os.path.join(pic_dir,ext))
        for pic in pic_list:
            if ignore not in pic:
                os.remove(pic)

def execCmd(cmd):
    
    #执行ffmpeg转换h264编码
    r = os.popen(cmd)
    text = r.read().strip()
    r.close()
    return text

#GPU特有   
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.7

if __name__ == '__main__':
    cap = cv2.VideoCapture('rtsp://admin:QACXNT@10.0.1.77:554/h264/ch1/main/av_stream')
#    cap = cv2.VideoCapture('rtmp://rtmp.open.ys7.com/openlive/f01c46bc25e3425390a6965838e7cdfe')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.info('摄像头未开启！')
        raise NameError
    elif fps < 100:
        fps_num = int(fps)
    else:
        fps_num = int(fps/10000)
    #控制录屏变量
    _numFrames = 0
    video_name = 'szewec'
    video_local_addr_pre = 'D:/nginx-rtmp/html/vod/'
    video_http_addr_pre = 'http://10.0.1.63:8088/vod/'
    pic_local_addr_pre = 'D:/nginx-rtmp/html/pic/'
    pic_http_addr_pre = 'http://10.0.1.63:8088/pic/'
    
    # rtmp video streaming
    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
    
    try:
        conn = connect_to_pgsql(PATH_TO_PGSQL)
        logger.info('Connect to PGSQL success!')
    except Exception as e:
        logger.error('PGSQL connect error:',e) 
    #电子围栏ID
    ID = 2
    row = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            while True:
                ret,image = cap.read()
                t = time.time()
                if ret == True:
                    
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    
                    #每10秒去数据库提取一次监控区域信息
                    if not time.gmtime().tm_sec%10 and int(time.time()*1000)%1000 < 100:
                        row = get_fence_area(conn,str(ID),'1')

                    #格式化监控区域信息
                    if len(row) == 0:
                        detect_area = False
                    else:
                        detect_area = []
                        for i in range(len(row)):
                            fpoint,fstatus = row[i][1],row[i][2]
                            fpoint = [int(i) for i in fpoint[1:-1].split(',')]
                            detect_list = np.array(fpoint).reshape(-1,2)
                            detect_area.append(detect_list)

                    #识别标记
                    [_,result] = vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                        detect_area=detect_area)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    #危险信息录屏
                    if result is True:
                        if (_numFrames == 0 and video_name == 'szewec'):
                            logger.info('进入警告区域开始录屏:')
                            #视频参数
                            name = get_format_time()
                            video_name = name + '.avi'
                            video_local_addr = video_local_addr_pre + name + '.avi'
                            video_http_addr = video_http_addr_pre + name + '.flv'
                            out = cv2.VideoWriter(video_local_addr, fourcc,fps_num, size)
                            #图片参数
                            pic_local_addr = pic_local_addr_pre + name + '.jpg'
                            pic_http_addr = pic_http_addr_pre + name + '.jpg'
                            cv2.imwrite(pic_local_addr,image_np)
                    if (_numFrames/fps_num < 30 and not video_name == 'szewec'):
                        out.write(image_np)
                        _numFrames += 1
                    else:
                        if (_numFrames/fps_num >= 30):
                            logger.info('录屏结束:')
                            cmd = 'ffmpeg -i {} -vcodec h264 {}'.format(video_local_addr,video_local_addr.split('.')[0]+'.flv')
                            r = sp.Popen(cmd, shell=True)
                            detect_result_to_pgsql('电子围栏', '2', pic_http_addr , video_http_addr, conn)
                            out.release()
                            _numFrames = 0
                            video_name = 'szewec'
                    #每60秒更新一次电子围栏信息
                    if not type(detect_area) == type(False):
                        for area in detect_area:
                            cv2.polylines(image_np,[area.reshape(-1,1,2)],True,(0,255,255),2)
                    
                    if not time.gmtime().tm_sec%60 and int(time.time()*1000)%1000 < 100:
                        #删除之前的围栏图片
                        pic_suffix = str(uuid.uuid4())[:4]
                        pic_local_addr = pic_local_addr_pre + 'camera_2_{}.jpg'.format(pic_suffix)
                        pic_http_addr = pic_http_addr_pre + 'camera_2_{}.jpg'.format(pic_suffix)
                        cv2.imwrite(pic_local_addr,image_np)
                        del_pic(pic_local_addr_pre,ext='camera_2*.jpg',ignore=pic_suffix)
                        update_fence_area(conn,pic_http_addr,'2')
                    else:
                        pass
                    #rtmp流写入
                    try:
                        proc.stdin.write(image_np.tobytes())
                    except Exception as e:
                        proc.stdin.close()
                        proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
                    cv2.imshow("frame",image_np)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    logger.info('结束时间：',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    break
        out.release()
        conn.close()
        proc.stdin.close()
        cap.release()
        cv2.destroyAllWindows()