from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView

from .od_core import ai_sight, base64_numpy_conversion

# Put your path to models and labels here, just for now.
PATH_TO_MODEL = ""
PATH_TO_LABELS = ""

class Image(APIView):

	renderer_classes = (JSONRenderer, )

	def __init__(self):
		ai_sight.__init__(PATH_TO_MODEL, PATH_TO_LABELS, 90)


	@csrf_exempt
	def post(self, request):

		data = JSONParser().parse(request)
		img_np = base64_numpy_conversion.base64_image_into_numpy_array(data["img"])
		np_img_with_boxes = ai_sight.draw_objects_boxes(img_np)
		img_b64 = base64_numpy_conversion.numpy_array_to_base64(np_img_with_boxes)

		response = {'img': img_b64}

		return Response(response)
