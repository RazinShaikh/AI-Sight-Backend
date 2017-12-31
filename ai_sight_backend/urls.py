from django.conf.urls import url
from django.contrib import admin
from object_detection.views import Img, Img2

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^detect/', Img.as_view()),
    url(r'^img2/', Img2.as_view()),
]
