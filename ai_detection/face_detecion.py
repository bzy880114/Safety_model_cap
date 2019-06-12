# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:25:38 2019

@author: sgs4167
"""
import numpy as np
import cv2
import dlib
import os

    
def face_detect():
    image = cv2.imread(r'D:\ProgramData\Spyder_Document\Safety_model_cap\fire_detection\uploads\10.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    model_dir = os.path.join(r'D:\ProgramData\Spyder_Document\Safety_model_cap\fire_detection',
                             r'fine_model\haarcascades\haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(model_dir)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.25, minNeighbors=3, minSize = (5, 5))
    for (x, y, w, h) in faces:
        cv2.rectangle(image,(x, y),(x+w, y+h),(0,255,0),2)
    cv2.namedWindow("img", 2)      # #图片窗口可调节大小
    cv2.imshow("img", image)       #显示图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def shape_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./fine_model/shape_predictor_68_face_landmarks.dat')
    
    # cv2读取图像
    img = cv2.imread(r"D:\ProgramData\Spyder_Document\Safety_model_cap\fire_detection\uploads\000.jpg")
    
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 2)

    for (i, rect) in enumerate(rects):
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #画出68个关键点
#        for (x, y) in shape:
#            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
#    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def simple_face_detect():
    detector = dlib.get_frontal_face_detector()
    image_to_check = cv2.imread(r"D:\ProgramData\Spyder_Document\Safety_model_cap\fire_detection\uploads\9.jpg")
    img_gray = cv2.cvtColor(image_to_check, cv2.COLOR_RGB2GRAY)        # 取灰度

    rects = detector(img_gray, 0)       # 进行检测
    print("Number of faces detected: {},rects: {}".format(len(rects),rects))     # 打印检测到的人脸数

    # 遍历返回的结果
    # 返回的结果是一个mmod_rectangles对象。这个对象包含有2个成员变量：dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度。

    face = rects[0]
    print("Detection Left: {} Top: {} Right: {} Bottom: {}".format(face.left(), face.top(), face.right(), face.bottom()))

#    cv2.rectangle()画出矩形,参数1：图像，参数2：矩形左上角坐标，参数3：矩形右下角坐标，参数4：画线对应的rgb颜色，参数5：线的宽度
    cv2.rectangle(image_to_check, (face.left(),face.top()), (face.right(),face.bottom()), (0,0,255),2)

#    cv2.namedWindow("img", 2)      # #图片窗口可调节大小
    cv2.imshow("img", image_to_check)       #显示图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    choice = 3
    if choice == 1:
        face_detect()
    elif choice == 2:
        shape_predictor()
    else:
        simple_face_detect()
    
    
    