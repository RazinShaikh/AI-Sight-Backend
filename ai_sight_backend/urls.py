from django.conf.urls import url
from django.contrib import admin
from object_detection.views import Img

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^img/', Img.as_view(), name="img"),
]
