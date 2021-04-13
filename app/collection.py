import cv2
import cvlib as cv

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
 
sample_num = 0    
captured_num = 0
    
# loop through frames
while webcam.isOpened():
    
    # read frame from webcam 
    status, frame = webcam.read()
    sample_num = sample_num + 1
    
    if not status:
        break
 
    # 이미지 내 얼굴 검출
    face, confidence = cv.detect_face(frame)
    
    print(face)
    print(confidence)
 
    # loop through detected faces
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
 
 
        if sample_num % 8  == 0:
            captured_num = captured_num + 1
            face_in_img = frame[startY:endY, startX:endX, :]
            face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./faces/user'+str(captured_num)+'.jpg', face_in_img)
            cv2.putText(face_in_img, str(captured_num), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
 
 
    # display output
    cv2.imshow("captured frames", frame)        
    
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()  