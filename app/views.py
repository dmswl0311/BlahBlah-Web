from django.shortcuts import render
from django.http.response import StreamingHttpResponse
# from app.camera import VideoCamera,VideoCameraImage,VideoCameraImage_Sad,VideoCameraImage_Birthday,VideoCameraImage_Crown
from app.detect_camera import VideoCamera2,VideoCameraImageSmile,VideoCameraImageSad,VideoCameraImageBirthday,VideoCameraImageCrown,VideoCollection,AppVideoCamera,Do
from django.conf import settings
from pathlib import Path
import glob
import os 
from app.my_modify import mModify
from django.http import HttpResponse
from time import sleep
import time

def index(request):
    return render(request, 'blog.html')

def blurring(request):
    return render(request,'blurring.html')

def decoration(request):
    url=request.GET.get('image_url')
    context={'urls':url}
    return render(request,'decoration.html',context) 

def gen(detect_camera):
    while True:
        frame = detect_camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera2()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_smile(request):
	return StreamingHttpResponse(gen(VideoCameraImageSmile()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_sad(request):
	return StreamingHttpResponse(gen(VideoCameraImageSad()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_birthday(request):
	return StreamingHttpResponse(gen(VideoCameraImageBirthday()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_crown(request):
	return StreamingHttpResponse(gen(VideoCameraImageCrown()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_collection(request):
    # faces에 있는 이미지 삭제 후 colletion 실행
    [os.remove(f) for f in glob.glob(os.path.join(settings.BASE_DIR,'app/faces/*.jpg'))]
    return StreamingHttpResponse(gen(VideoCollection()),content_type='multipart/x-mixed-replace; boundary=frame')

def collection(request):
    return render(request,'collection.html')

def my_modify(request):
    mModify()
    return render(request,'processing.html')

def processing(request):
    return render(request,'processing.html')

def App_processing(request):
    return render(request,'App_processing.html')

def video_app(request):
	return StreamingHttpResponse(gen(AppVideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_capture(request):
    Do()
    sleep(1000)
    return StreamingHttpResponse(gen(VideoCamera2()),content_type='multipart/x-mixed-replace; boundary=frame')