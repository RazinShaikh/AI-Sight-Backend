import base64
import numpy as np
from PIL import Image
import io

def base64_image_into_numpy_array(base64_image):
	decoded_img = base64.b64decode(base64_image)
	image_data = Image.open(io.BytesIO(decoded_img))
	(im_width, im_height) = image_data.size
	return np.array(image_data.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


def numpy_array_to_base64(numpy_image):
	image_data = Image.fromarray(numpy_image)
	buffered = io.BytesIO()
	image_data.save(buffered, format="PNG")
	img_b64 = base64.b64encode(buffered.getvalue())

	return img_b64