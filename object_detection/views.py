from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.parsers import JSONParser
from rest_framework.views import APIView

import base64
import numpy as np
from PIL import Image
import io

class detection(APIView):
	
	@csrf_exempt
	def post(self, request):
		data = JSONParser().parse(request)
		img_np = load_base64_image_into_numpy_array(data["img"])
		return HttpResponse(img_np)

def load_base64_image_into_numpy_array(base64_image):
	decoded_img = base64.b64decode(base64_image)
	image_data = Image.open(io.BytesIO(decoded_img))
	(im_width, im_height) = image_data.size
	return np.array(image_data.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)