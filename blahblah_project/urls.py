from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('video_image', views.video_image, name='video_image'),
    path('image/', views.image, name='image'),
    path('blur/', views.blur, name='blur'),
]
