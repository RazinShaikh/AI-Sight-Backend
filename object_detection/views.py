from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.parsers import JSONParser
from rest_framework.views import APIView

from .od_core import ai_sight, b64_to_np

class image(APIView):
	
	@csrf_exempt
	def post(self, request):
		data = JSONParser().parse(request)
		img_np = b64_to_np.load_base64_image_into_numpy_array(data["img"])
		tmpresult = tmp(img_np)
		return HttpResponse(tmpresult)


def tmp(image_np):
	testInstance = ai_sight.detection("C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\backend\\object_detection\\od_core\\pretrained models\\rfcn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb",
										"C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\backend\\object_detection\\od_core\\labels\\mscoco_label_map.pbtxt"
										)
	return testInstance.detect(image_np)

