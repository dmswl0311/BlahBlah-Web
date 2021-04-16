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
from os import listdir
from os.path import isfile, join
from time import sleep
import time
import datetime

data_path = os.path.join(settings.BASE_DIR,'app/faces/')
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# 이모지 붙일 때, 이미지 경로
src1=cv2.imread(os.path.join(settings.BASE_DIR,'img/smile.png'),-1)
src2=cv2.imread(os.path.join(settings.BASE_DIR,'img/sad.png'),-1)
src3=cv2.imread(os.path.join(settings.BASE_DIR,'img/birthday.png'),-1)
src4=cv2.imread(os.path.join(settings.BASE_DIR,'img/crown.png'),-1)

# ip cam url
url='rtsp://192.168.0.19:8080/h264_ulaw.sdp'

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# smile, sad 이모지 붙이는 함수 ===================================================
def transparent_overlay(src ,overlay ,pos=(0,0) ,scale=1):
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
    
    y, x = pos[0], pos[1]   

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) # read the alpha channel 
            src[x + i][y + j] = alpha*overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

# birthday, crown이모지 붙이는 함수 ===================================================
def transparent_overlay_birthday(src ,overlay ,pos=(0,0) ,scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  
    rows, cols, _ = src.shape  

    y, x = pos[0], pos[1]-(h+2) 

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) 
            src[x + i][y + j] = alpha*overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi


global a # frame
a=0

def Do():
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    # print("a(frame) 값: ",a)
    # print(str(os.path.join(settings.BASE_DIR,'./capture/')))
    cv2.imwrite(str(os.path.join(settings.BASE_DIR,'./capture/'))+str(now)+".jpg", a)

class VideoCamera2(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        global a

        ret, frame = self.webcam.read()
    
        if not ret:
            print("Could not read frame")
            exit()

        face, confidence = cv.detect_face(frame)

        try:
            # loop through detected faces
            if len(confidence)>=1:
                for idx, f in enumerate(face):
                    
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                        # Y = startY - 10 if startY - 10 > 10 else startY + 10
                        # text = str(confidence)+'% Confidence it is target'
                        # cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        roi = cv2.GaussianBlur(roi, (29, 29), 10) # 블러(모자이크) 처리
                        frame[startY:endY, startX:endX] = roi 
                a=frame
            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass

        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()


class VideoCameraImageSmile(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            print("Could not read frame")
            exit()
            
        face, confidence = cv.detect_face(frame)

        try:
            if len(confidence)>=1:
            # loop through detected faces
                for idx, f in enumerate(face):
                    
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        frame[startY:endY, startX:endX] = roi 
                        src = cv2.resize(src1, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                        frame = transparent_overlay(frame, src, (startX, startY))

            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass
        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImageSad(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            print("Could not read frame")
            exit()
            
        face, confidence = cv.detect_face(frame)

        try:
            if len(confidence)>=1:
            # loop through detected faces
                for idx, f in enumerate(face):
                    
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        frame[startY:endY, startX:endX] = roi 
                        src = cv2.resize(src2, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                        frame = transparent_overlay(frame, src, (startX, startY))
            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass
        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImageBirthday(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            print("Could not read frame")
            exit()
            
        face, confidence = cv.detect_face(frame)

        try:
            if len(confidence)>=1:
            # loop through detected faces
                for idx, f in enumerate(face):
                    
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        frame[startY:endY, startX:endX] = roi 
                        src = cv2.resize(src3, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                        frame = transparent_overlay_birthday(frame, src, (startX, startY))
            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass
        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

class VideoCameraImageCrown(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            print("Could not read frame")
            exit()
            
        face, confidence = cv.detect_face(frame)

        try:
            if len(confidence)>=1:
            # loop through detected faces
                for idx, f in enumerate(face):
                    
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        frame[startY:endY, startX:endX] = roi 
                        src = cv2.resize(src4, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                        frame = transparent_overlay_birthday(frame, src, (startX, startY))
            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass
        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

sample_num = 0    
captured_num = 0
face_num=0

# Face Colletion Class ====================================

class VideoCollection(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        global sample_num
        global captured_num
        global face_num
        # sample_num = 0    
        # captured_num = 0

        ret, frame = self.webcam.read()
        # sample_num = sample_num + 1
        sample_num = sample_num + 1
        if not ret:
            print("Could not read frame")
            exit()
            
        face, confidence = cv.detect_face(frame)

        if len(confidence)==1:
            # loop through detected faces
            for idx, f in enumerate(face):
                        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                if sample_num % 8  == 0:
                    # captured_num = captured_num + 1
                    captured_num = captured_num + 1
                    print("captured_num= ",captured_num)
                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                        
                    cv2.imwrite('app/faces/user'+str(captured_num)+'.jpg', face_in_img)
                    face_in_img_flip = cv2.flip(face_in_img, 1)
                    cv2.imwrite('app/faces/user'+str(captured_num+100)+'.jpg', face_in_img_flip)

                # 100장 학습
                if captured_num==100:
                    # cv2.putText(frame, "colletion complete!", (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    raise StopIteration
                else:
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    cv2.putText(frame, str(captured_num+1), (20,30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

        elif len(confidence)>1:
            image_url=cv2.imread(os.path.join(settings.BASE_DIR,'img/overface.PNG'),-1)
            frame = colletion_overlay(frame, image_url, (150, 400))
        else:
            image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
            frame = colletion_overlay(frame, image_url2, (150, 400))
            pass

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()


def colletion_overlay(src ,overlay ,pos=(0,0) ,scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """

    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    
    y, x = pos[0], pos[1]   

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) # read the alpha channel 
            src[x + i][y + j] = alpha*overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

current_time=0
prev_time=0
fps=0

class AppVideoCamera(object):
    def __init__(self):
        global url
        self.webcam = cv2.VideoCapture(url)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        global current_time
        global prev_time
        global fps

        ret, frame = self.webcam.read()

        current_time = time.time() - prev_time

        if not ret:
            print("Could not read frame")
            exit()

        face, confidence = cv.detect_face(frame)

        try:
            if len(confidence)>=1:
            # loop through detected faces
                for idx, f in enumerate(face):
                        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        roi = cv2.GaussianBlur(roi, (29, 29), 10) # 블러(모자이크) 처리
                        frame[startY:endY, startX:endX] = roi 
                    prev_time = time.time()
            else:
                image_url2=cv2.imread(os.path.join(settings.BASE_DIR,'img/noface.PNG'),-1)
                frame = colletion_overlay(frame, image_url2, (150, 400))
                pass
        except:
            # cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        fps +=1

        ret,jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()