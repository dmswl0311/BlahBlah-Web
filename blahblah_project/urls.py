from django.contrib import admin
from django.urls import path
from app import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('video_image_smile', views.video_image_smile, name='video_image_smile'),
    path('video_image_sad', views.video_image_sad, name='video_image_sad'),
    path('video_image_birthday', views.video_image_birthday, name='video_image_birthday'),
    path('video_image_crown', views.video_image_crown, name='video_image_crown'),
    path('video_collection', views.video_collection, name='video_collection'),
    path('decoration/', views.decoration, name='decoration'),
    path('blurring/', views.blurring, name='blurring'),
    path('collection/', views.collection, name='collection'),
]

# 이미지 URL 설정
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 