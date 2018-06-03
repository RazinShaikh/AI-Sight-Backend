from django.conf.urls import url
from django.contrib import admin
from object_detection.views import Img
from object_detection.views import Text

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^img/', Img.as_view(), name="img"),
    url(r'^text/', Text.as_view(), name="text"),
]
