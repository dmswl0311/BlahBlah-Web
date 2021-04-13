from django.contrib import admin
from django.urls import path
from app import views
# index는 대문, blog는 게시판
# from views import index, posting
# 이미지를 업로드하자
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    # path('<int:pk>/', views.posting, name='posting'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('video_image', views.video_image, name='video_image'),
    path('decoration/', views.decoration, name='decoration'),
    path('blurring/', views.blurring, name='blurring'),
]

# 이미지 URL 설정
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 