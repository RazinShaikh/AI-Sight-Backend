import base64
import io
import numpy as np

from PIL import Image


def base64_image_into_numpy_array(base64_image):
    decoded_img = base64.b64decode(base64_image)
    image_data = Image.open(io.BytesIO(decoded_img))
    (im_width, im_height) = image_data.size
    return np.array(image_data).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
