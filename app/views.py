from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from app.camera import VideoCamera
from app.camera import VideoCameraImage
from tensorflow.keras.models import load_model
from pathlib import Path
# View에 Model(Post 게시글) 가져오기
from .models import Post

# index.html 페이지를 부르는 index 함수
def index(request):
    # 모든 Post를 가져와 postlist에 저장
    postlist = Post.objects.all()
    # blog.html 페이지를 열 때, 모든 Post인 postlist도 같이 가져옴 
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