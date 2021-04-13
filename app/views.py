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
    return render(request, 'blog.html', {'postlist':postlist})

# blog의 게시글(posting)을 부르는 posting 함수
def posting(request, pk):
    # 게시글(Post) 중 pk(primary_key)를 이용해 하나의 게시글(post)를 검색
    post = Post.objects.get(pk=pk)
    # posting.html 페이지를 열 때, 찾아낸 게시글(post)을 post라는 이름으로 가져옴
    
    if pk == 1:
        return render(request,'blurring.html', {'post':post})
    else :
        return render(request,'decoration.html', {'post':post})

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def video_image(request):
	return StreamingHttpResponse(gen(VideoCameraImage()),content_type='multipart/x-mixed-replace; boundary=frame')

def blur(request):
    return render(request,'blur.html')

def image(request):
    return render(request,'image.html')