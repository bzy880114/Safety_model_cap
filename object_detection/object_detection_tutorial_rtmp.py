# -*- coding: utf-8 -*-
"""
Created on :2018/8/29 18:28

@author: sgs4176
"""
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import sys
import subprocess as sp
import psycopg2
import configparser
import datetime

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
 
 
# # This is needed to display the images.
# %matplotlib inline
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
 
# from utils import label_map_util
# from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
 
# What model to download.
MODEL_NAME = 'frozen_inference_graph_ssd_m_v1_2'
#MODEL_NAME = 'frozen_inference_graph_ssd_m_v1'
#MODEL_NAME = 'frozen_inference_graph_frcnn_inception'
#MODEL_NAME = 'ssd_mobilenet_v1_coco'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 2

# pgsql config
PATH_TO_PGSQL = os.path.join('data', 'pgsql.config')

# rtmp address
rtmpUrl = 'rtmp://10.0.1.63:1935/live/stream'
 
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
category_index[1]['name']=u'无安全帽'
category_index[2]['name']=u'安全帽'
#category_index[1]['name']=u'行人'
#category_index[76]['name']=u'键盘'
#category_index[64]['name']=u'盆摘'
#category_index[86]['name']=u'花瓶'
#category_index[47]['name']=u'水杯'
#category_index[77]['name']=u'手机'
#category_index[62]['name']=u'椅子'
#category_index[72]['name']=u'显示器'


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
def detect_result_to_pgsql(result, category_index, video_id, conn):
    classes = np.squeeze(result[0])
    scores  = np.squeeze(result[1])
    cur = conn.cursor()
    
    num = scores[scores>0.5].shape[0]
    class_id = classes[scores>0.5]
    for i in range(num):
        if class_id[i] == 2:
            timeStamp = time.time()
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                cur.execute("insert into t_mon_message VALUES (%s, %s, '0', %s, %s);" , (int(round(timeStamp * 1000)), str(class_id[i]), nowTime, video_id))
            except Exception as e:
                print('excute sql error!')
    conn.commit()
    cur.close()

# video detect 
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.  0
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1)
    return (image_np,classes,scores)


def worker(input_q, output_q, output_class, config):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph, config=config)

    fps = FPS().start()

    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (image_np,classes,scores) = detect_objects(frame_rgb, sess, detection_graph)
        output_class.put([classes,scores])
        output_q.put(image_np)
    fps.stop()
    sess.close()

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default='rtmp://rtmp.open.ys7.com/openlive/af3e3230f4da4201bb71a5b0da68e034', help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=4, help='Size of the queue.')
    parser.add_argument('-gpu-frac', '--gpu_memory_fraction', dest='gpu_memory_fraction', type=float,
                        default=0.4, help='fraction of the gpu_memory.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

#GPU特有   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction



    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    output_class = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q, output_class, config))
    
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start() 
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print('摄像头未开启！')
        raise NameError
    elif fps < 100:
        fps_num = int(fps)
    else:
        fps_num = int(fps/10000)
    
    # rtmp video streaming
    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
    
    fps = FPS().start()
#    try:
#        conn = connect_to_pgsql(PATH_TO_PGSQL)
#        print('Connect to PGSQL success!')
#    except Exception as e:
#        print('PGSQL connect error:',e) 
    # 视频存档
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('./out.avi', fourcc, fps_num, size)

    while True:
        try:
            frame = video_capture.read()
        except Exception as e :
            print('Frame is None,the message is :%s' % e)

        if frame is not None:
            input_q.put(frame)
            t = time.time()
            
            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            result = output_class.get()
#            if (fps._numFrames%50 == 0):
#                detect_result_to_pgsql(result, category_index, '1', conn)
            proc.stdin.write(output_rgb.tobytes())
            
            cv2.imshow('Video', output_rgb)

            fps.update()
            print('[INFO] elapsed time: {:.4f}'.format(time.time() - t))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            video_capture.stop()
            video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
            
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))     

#    conn.close()
    proc.stdin.close()
    pool.terminate()
    pool.join()
    video_capture.stop()
    cv2.destroyAllWindows()