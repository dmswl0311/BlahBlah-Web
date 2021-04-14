from tkinter import *
from tkinter import filedialog
import os
from django.conf import settings
import cvlib as cv
import cv2
import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join

data_path = os.path.join(settings.BASE_DIR,'app/faces/')
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

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

def mModify():
    root = Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()
    root.filename =  filedialog.askopenfilename(initialdir = "/", title = "Select file")
   
    print("#######################")
    print(root.filename)
    f1,f2 =os.path.splitext(root.filename)
    print(f2)
    flag = '0'
    if f2 == '.png' or f2 == '.jpg' or f2 == '.jpeg':
        flag = '1'
    elif f2 == '.avi':
        flag = '2'
    elif f2 == '.mp4':
        flag = '3'
    now = datetime.datetime.now().strftime("-%d-%H-%M-%S")
    if len(str(root.filename)) < 1:
        print("File error")
        return
    
    if flag == '1':
        print("Flag 1")
        frame = cv2.imread(str(root.filename))
        face, confidence = cv.detect_face(frame)
        try:
            
            print("Try 1")
            # loop through detected faces
            for idx, f in enumerate(face):
                
                print("Face 1")
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                face_in_img = frame[startY:endY, startX:endX, :]
                face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                result = model.predict(face_in_img)
                confidence = int(100*(1-(result[1])/300))

                if confidence > 70:
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    #Y = startY - 10 if startY - 10 > 10 else startY + 10
                    #text = str(confidence)+'% Confidence it is target'
                    #cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                else:
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    roi = cv2.GaussianBlur(roi, (29, 29), 10) # 블러(모자이크) 처리
                    frame[startY:endY, startX:endX] = roi
            cv2.imshow("modify",frame)
            cv2.imwrite(f1 + str(now) + ".png", frame)
        except:
            print("Face Not Found")
            cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(f1 + str(now) + ".png", frame)
           
    elif flag == '2' or flag == '3':
        print("Flag 2")
        webcam = cv2.VideoCapture(str(root.filename))
        SetCodec = False
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()
            #####
     
        while webcam.isOpened():
            status, frame = webcam.read()
            if not status:
                video.release()
                return
            if SetCodec == False:
                if flag == '2':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video = cv2.VideoWriter(f1 + str(now) + ".avi", fourcc, 20.0, (frame.shape[1],frame.shape[0]))
                    SetCodec = True
                elif flag == '3':
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    video = cv2.VideoWriter(f1 + str(now) + ".mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    SetCodec = True
            face, confidence = cv.detect_face(frame)
           
            try:
                print("Try 2")
                # loop through detected faces
                for idx, f in enumerate(face):
                    
                    print("Face 2")
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_in_img = frame[startY:endY, startX:endX, :]
                    face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face_in_img)
                    confidence = int(100*(1-(result[1])/300))

                    if confidence > 70:
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                        #Y = startY - 10 if startY - 10 > 10 else startY + 10
                        #text = str(confidence)+'% Confidence it is target'
                        #cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    else:
                        roi = frame[startY:endY, startX:endX] # 관심영역 지정
                        roi = cv2.GaussianBlur(roi, (29, 29), 10) # 블러(모자이크) 처리
                        frame[startY:endY, startX:endX] = roi 
                        
                print("Write 2")
                video.write(frame)
            except:
                print("Face Not Found")
                cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                video.write(frame)
        video.release()
    root.mainloop()
        
    