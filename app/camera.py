# import necessary packages
import cv2,os
import imutils
from imutils.video import VideoStream
import cvlib as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
from django.conf import settings
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

model = load_model(os.path.join(settings.BASE_DIR,'face_detector/saved_model.h5'))
# 이미지 파일 경로, 바꿔야됨
src1=cv2.imread(os.path.join(settings.BASE_DIR,'img/smile.png'),-1)
src2=cv2.imread(os.path.join(settings.BASE_DIR,'img/sad.png'),-1)
src3=cv2.imread(os.path.join(settings.BASE_DIR,'img/birthday.png'),-1)
src4=cv2.imread(os.path.join(settings.BASE_DIR,'img/crown.png'),-1)
flag=0

def transparent_overlay(src ,overlay ,pos=(0,0) ,scale=1,flag=0):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    print(flag)
    if flag==1: #birthday
        y, x = pos[0], pos[1]-(h+2) 
    elif flag==2:
        y, x = pos[0], pos[1]-h-2
    else:
        y, x = pos[0], pos[1]   

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) # read the alpha channel 
            src[x + i][y + j] = alpha*overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        status, frame = self.video.read()
        # apply face detection
        face, confidence = cv.detect_face(frame)

        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                
                face_region = frame[startY:endY, startX:endX]
                
                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                
                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                prediction = model.predict(x)

                if prediction < 0.1: # 타켓 판별
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                else: # 논타켓 판별
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    roi = cv2.GaussianBlur(roi, (0, 0), 3) # 블러(모자이크) 처리
                    frame[startY:endY, startX:endX] = roi 
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImage(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        flag=0
        status, frame = self.video.read()

        face, confidence = cv.detect_face(frame)
        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                
                face_region = frame[startY:endY, startX:endX]
                
                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                
                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                prediction = model.predict(x)

                if prediction < 0.1: # 타켓 판별
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                else: # 논타켓 판별
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    frame[startY:endY, startX:endX] = roi 
                    src = cv2.resize(src1, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                    frame = transparent_overlay(frame, src, (startX, startY),flag)
                    
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImage_Sad(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        flag=0
        status, frame = self.video.read()

        face, confidence = cv.detect_face(frame)
        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                
                face_region = frame[startY:endY, startX:endX]
                
                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                
                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                prediction = model.predict(x)

                if prediction < 0.1: # 타켓 판별
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                else: # 논타켓 판별
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    frame[startY:endY, startX:endX] = roi 
                    src = cv2.resize(src2, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                    frame = transparent_overlay(frame, src, (startX, startY),flag)
                    
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImage_Birthday(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        status, frame = self.video.read()

        face, confidence = cv.detect_face(frame)
        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                
                face_region = frame[startY:endY, startX:endX]
                
                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                
                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                prediction = model.predict(x)

                if prediction < 0.1: # 타켓 판별
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                else: # 논타켓 판별
                    flag=1
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    frame[startY:endY, startX:endX] = roi 
                    src = cv2.resize(src3, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                    frame = transparent_overlay(frame, src, (startX, startY),flag)
                    
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImage_Crown(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        status, frame = self.video.read()

        face, confidence = cv.detect_face(frame)
        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
                
                face_region = frame[startY:endY, startX:endX]
                
                face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
                
                x = img_to_array(face_region1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                prediction = model.predict(x)

                if prediction < 0.1: # 타켓 판별
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                else: # 논타켓 판별
                    flag=2
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    frame[startY:endY, startX:endX] = roi 
                    src = cv2.resize(src4, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                    frame = transparent_overlay(frame, src, (startX, startY),flag)
                    
        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()