# -*- coding: utf-8 -*-

import numpy as np
import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import subprocess as sp
import sys
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("..")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

MODEL_NAME = 'frozen_inference_graph_ssd_m_v1_5'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = r'D:\ProgramData\Spyder_Document\Safety_model_cap\object_detection\frozen_inference_graph_ssd_m_v1_5\frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r'D:\ProgramData\Spyder_Document\Safety_model_cap\object_detection\data\label_map.pbtxt'

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


def contrast_brightness(image, c, b):  #其中c为对比度，b为每个像素加上的值（调节亮度）
    blank = np.zeros(image.shape, image.dtype)   #创建一张与原图像大小及通道数都相同的黑色图像
    dst = cv2.addWeighted(image, c, blank, 1-c, b) #c为加权值，b为每个像素所加的像素值
    ret, dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    return dst


def drawfire_bak(image,fireimage):
    display_str = str('火焰预警')
    image_np = Image.fromarray(image)
    draw = ImageDraw.Draw(image_np)
    _,contours, hierarchy = cv2.findContours(fireimage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)     
    v_box = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if len(contours) < 4:
            break
        x,y,w,h = cv2.boundingRect(cnt)
        if area>1:
            v_box.append([x,y])
            v_box.append([x+w, y+h])

    if (len(v_box)>0):
        v_box = np.array(v_box)
        minvx,minvy = np.amin(v_box,axis=0)
        maxvx, maxvy = np.amax(v_box,axis=0)
                
    if (len(v_box)>0 ):
        font = ImageFont.truetype('simhei.ttf', 20,encoding='utf-8')
        display_str_height = font.getsize(display_str)[1]
        display_str_heights = (1 + 2 * 0.05) * display_str_height
        if minvy > display_str_heights:
            text_bottom = minvy
        else:
            text_bottom = maxvy
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                [(minvx, text_bottom - text_height - 2 * margin), 
                 (minvx + text_width,text_bottom)],fill='blue')
        draw.text(
                (minvx + margin, text_bottom - text_height - margin),
                display_str,fill='yellow',font=font)
        image = np.array(image_np)
        cv2.rectangle(image, (minvx,minvy), (maxvx, maxvy), (0, 255, 0), 1)
    return image


def fire_model(url, rtmpUrl):
    cap = cv2.VideoCapture(url)
    width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    redThre = 135
    saturationTh = 55
    ctrl = 3

#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise NameError
    elif fps < 100:
        fps_num = int(fps)
    else:
        fps_num = int(fps/10000)
    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)

## 二帧差分
    if(ctrl==2):
        frameNum = 0
        while(True):
            ret, frame = cap.read()
            frameNum += 1
            if ret == True:   
                tempframe = frame    
                if(frameNum==1):
                    previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
                if(frameNum>=2):
                    currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)        
                    currentframe = cv2.absdiff(currentframe,previousframe)
#                    median = cv2.medianBlur(currentframe,3)
                    ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(threshold_frame)
                    gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)
    
                    B = frame[:, :, 0]
                    G = frame[:, :, 1]
                    R = frame[:, :, 2]
                    minValue = np.array(np.where(R <= G, np.where(G < B, R, 
                                                                  np.where(R < B, R, B)), np.where(G < B, G, B)))
                    S = 1 - 3.0 * minValue / (R + G + B + 1)
#                    fireImg = np.array(np.where(R > redThre, 
#                                                np.where(R > G, 
#                                                         np.where(G > B, 
#                                                                  np.where(S > 0.2, 
#                                                                           np.where(S > (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0), 0))
                    fireImg = np.array(np.where(R > redThre, 
                                                np.where(R > G, 
                                                         np.where(G > B, 
                                                                  np.where(S > (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0))
                    
                    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
                    gray_fireImg[:, :, 0] = fireImg
                    gray_fireImg = cv2.GaussianBlur(gray_fireImg, (3, 3), 0)  
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    gauss_image = cv2.morphologyEx(gauss_image, cv2.MORPH_OPEN, kernel)
                    cv2.imshow("gauss_image",gauss_image)
                    
                    gray_fireImg = contrast_brightness(gray_fireImg, 5., 25)
                    cv2.imshow("gray_fireImg",gray_fireImg)
                    gray_fireImg = cv2.bitwise_and(gray_fireImg,gauss_image,mask=mask_inv)
            
                    image = drawfire_bak(frame, gray_fireImg)
                    cv2.imshow("img", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    #rtmp流写入
                    try:
                        proc.stdin.write(image.tobytes())
                    except Exception as e:
                        proc.stdin.close()
                        proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
                previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            else:
                break
        proc.stdin.close()
        cap.release()
        cv2.destroyAllWindows()
## 三帧差分法
    else:
        one_frame = np.zeros((height,width),dtype=np.uint8)
        two_frame = np.zeros((height,width),dtype=np.uint8)
        three_frame = np.zeros((height,width),dtype=np.uint8)
        while(True):
            ret, frame = cap.read()
            if ret == True: 
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                one_frame,two_frame,three_frame = two_frame,three_frame,frame_gray
                abs1 = cv2.absdiff(one_frame,two_frame)#相减
                _,thresh1 = cv2.threshold(abs1,20,255,cv2.THRESH_BINARY)#二值，大于20的为255，小于0
                abs2 =cv2.absdiff(two_frame,three_frame)
                _,thresh2 =cv2.threshold(abs2,20,255,cv2.THRESH_BINARY)

                binary =cv2.bitwise_and(thresh1,thresh2)

                B = frame[:, :, 0]
                G = frame[:, :, 1]
                R = frame[:, :, 2]
                minValue = np.array(np.where(R <= G, np.where(G < B, R, 
                                                              np.where(R < B, R, B)), np.where(G < B, G, B)))
                RGB_sum = R + G + B
                RGB_sum[RGB_sum == 0] = 1
                S = 1 - 3.0 * minValue / RGB_sum
#                    fireImg = np.array(np.where(R > redThre, 
#                                                np.where(R > G, 
#                                                         np.where(G > B, 
#                                                                  np.where(S > 0.2, 
#                                                                           np.where(S > (255 - R)*saturationTh/redThre, 255, 0), 0), 0), 0), 0))
                estimate = (255 - R)*saturationTh/redThre
                fireImg = np.array(np.where(R > redThre, 
                                            np.where(R > G, 
                                                     np.where(G > B, 
                                                              np.where(S > estimate, 255, 0), 0), 0), 0))
                
                gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
                gray_fireImg[:, :, 0] = fireImg
                gray_fireImg = cv2.GaussianBlur(gray_fireImg, (3, 3), 0)  
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                gray_fireImg = contrast_brightness(gray_fireImg, 5., 25)
                gray_fireImg = cv2.bitwise_and(gray_fireImg,binary)
        
                image = drawfire_bak(frame, gray_fireImg)
#                cv2.imshow("img", image)
#                if cv2.waitKey(1) & 0xFF == ord('q'):
#                    break
                #rtmp流写入
                try:
                    proc.stdin.write(image.tobytes())
                except Exception as e:
                    proc.stdin.close()
                    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
            else:
                break
        proc.stdin.close()
        cap.release()
        cv2.destroyAllWindows()
        
        
def safetycap_model_pic(url):
    image = cv2.imread(url)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
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
                line_thickness=2)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np
            
def safetycap_model_video(url, rtmpUrl):
    cap = cv2.VideoCapture(url)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = cap.get(cv2.CAP_PROP_FPS) # 30p/self
    if fps == 0:
        raise NameError
    elif fps < 100:
        fps_num = int(fps)
    else:
        fps_num = int(fps/10000)

    # rtmp video streaming
    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            while True:
                ret,image = cap.read()   
                if ret == True:
                    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
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
                        line_thickness=2)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    proc.stdin.write(image_np.tobytes())
#                    cv2.imshow('image_np',image_np)
#                    if cv2.waitKey(1) & 0xFF == ord('q'):
#                        break
                else:
                    break
        proc.stdin.close()
        cap.release()
        cv2.destroyAllWindows()
        
#if __name__ == '__main__':
#    res = safetycap_model_video(r'D:\ProgramData\Spyder_Document\Safety_model_cap\fire_detection\uploads\4997d2b39b56.avi','rtmp://10.0.1.63:1937/live/stream12')
#    cv2.imshow('img',res)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()