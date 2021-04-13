from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from app.camera import VideoCamera,VideoCameraImage,VideoCameraImage_Sad,VideoCameraImage_Birthday,VideoCameraImage_Crown
from tensorflow.keras.models import load_model
from pathlib import Path

def index(request):
    return render(request, 'blog.html')

def blurring(request):
    return render(request,'blurring.html')

def decoration(request):
    url=request.GET.get('image_url')
    context={'urls':url}
    return render(request,'decoration.html',context) 

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_smile(request):
	return StreamingHttpResponse(gen(VideoCameraImage()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_birthday(request):
	return StreamingHttpResponse(gen(VideoCameraImage_Birthday()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_crown(request):
	return StreamingHttpResponse(gen(VideoCameraImage_Crown()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image_sad(request):
	return StreamingHttpResponse(gen(VideoCameraImage_Sad()),content_type='multipart/x-mixed-replace; boundary=frame')