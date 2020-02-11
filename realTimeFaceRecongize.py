# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
# 加载人脸特征库
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml');
#抑制人脸识别的边框.opencv会在我的人脸附近输出多个边框.
#这两个变量用于记录最大值
NonMaximum_suppression_h = 0;
NonMaximum_suppression_w = 0;
portrait = np.array([0]);

while(True):
    ret, frame = cap.read(); # 读取一帧的图像
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);# 转灰
    #识别人脸
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.15, minNeighbors = 5, minSize = (5, 5)) # 检测人脸
    i = 0; 
    for(x, y, w, h) in faces:
        #得到最大值
        if NonMaximum_suppression_h < h or NonMaximum_suppression_w < w:
            NonMaximum_suppression_h = h;
            NonMaximum_suppression_w = w;
            i=i+1;
            if i>15:#计算满20就清零
                i=0;
        #达到了要求才可以输出
        if NonMaximum_suppression_h*0.7 < h and NonMaximum_suppression_w*0.7 < w:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0),2);# 用矩形圈出人脸
            portrait = frame[x-50:x+50,y-50:y+50];
            
    cv2.imshow('Face Recognition',frame);
    
    if portrait.shape[0] > 90:
        plt.figure("face")
        plt.imshow(portrait)
        plt.show()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release();
cv2.destroyAllWindows()

