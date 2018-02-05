from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

import io
import numpy as np
import os

from .od_core import ai_sight
from .od_core import base64_numpy_conversion
from ai_sight_backend.settings import BASE_DIR


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

PATH_TO_MODEL = os.path.join(BASE_DIR,
                             'object_detection',
                             'od_core',
                             'pretrained_models',
                             MODEL_NAME,
                             'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(BASE_DIR,
                              'object_detection',
                              'od_core',
                              'labels',
                              'mscoco_label_map.pbtxt')


class Img(APIView):

    renderer_classes = (JSONRenderer, )

    def __init__(self):
        ai_sight.__init__(PATH_TO_MODEL, PATH_TO_LABELS, 90)

    @csrf_exempt
    def post(self, request):

        data = JSONParser().parse(request)
        img_np = base64_numpy_conversion.base64_image_into_numpy_array(
            data["img"])
        np_img_with_boxes = ai_sight.draw_objects_boxes(img_np)
        img_b64 = base64_numpy_conversion.numpy_array_to_base64(
            np_img_with_boxes)

        response = {'img': img_b64}

        return Response(response)


class Img2(APIView):

    renderer_classes = (JSONRenderer, )

    def __init__(self):
        ai_sight.__init__(PATH_TO_MODEL, PATH_TO_LABELS, 90)

    @csrf_exempt
    def post(self, request):
        print("request received")
        image_data = Image.open(io.BytesIO(request.body))

        (im_width, im_height) = image_data.size
        img_np = np.array(image_data.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

        np_img_with_boxes = ai_sight.draw_objects_boxes(img_np)
        img_b64 = base64_numpy_conversion.numpy_array_to_base64(
            np_img_with_boxes)

        response = {'img': img_b64}

        return Response(response)


class Img3(APIView):
    renderer_classes = (JSONRenderer, )

    def __init__(self):
        ai_sight.__init__(PATH_TO_MODEL, PATH_TO_LABELS, 90)

    @csrf_exempt
    def post(self, request):
        print("request received")
        image_data = Image.open(io.BytesIO(request.body))

        (im_width, im_height) = image_data.size
        img_np = np.array(image_data.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

        boxes, scores, classes = ai_sight.get_detection_result(img_np)

        response = {'boxes': boxes, 'scores': scores, 'classes': classes}

        return Response(response)
