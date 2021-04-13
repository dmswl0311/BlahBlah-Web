from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from app.camera import VideoCamera
from app.camera import VideoCameraImage
from tensorflow.keras.models import load_model
from pathlib import Path

def index(request):
    return render(request, 'blog.html')

def blurring(request):
    return render(request,'blurring.html')

def decoration(request):
    return render(request,'decoration.html') 

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image(request):
	return StreamingHttpResponse(gen(VideoCameraImage()),content_type='multipart/x-mixed-replace; boundary=frame')

def blurring(request):
    return render(request,'blurring.html')

def decoration(request):
    return render(request,'decoration.html')