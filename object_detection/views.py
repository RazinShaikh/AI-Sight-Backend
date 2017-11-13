from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.parsers import JSONParser
from rest_framework.views import APIView

from .od_core import ai_sight, b64_to_np

# Put your path to models and labels here, just for now.
PATH_TO_MODEL = ""
PATH_TO_LABELS = ""

class Image(APIView):

	def __init__(self):
		ai_sight.__init__(PATH_TO_MODEL, PATH_TO_LABELS, 90)

	@csrf_exempt
	def post(self, request):
		data = JSONParser().parse(request)
		img_np = b64_to_np.load_base64_image_into_numpy_array(data["img"])
		result = ai_sight.detect(img_np)
		return HttpResponse(result)
