from django.conf.urls import url
from django.contrib import admin
from object_detection.views import image

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^detect/', image.as_view()),

]
