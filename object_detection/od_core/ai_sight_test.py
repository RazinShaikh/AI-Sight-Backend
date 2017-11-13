from PIL import Image
import unittest
import ai_sight
import numpy as np

class TestDetection(unittest.TestCase):
	
	@unittest.skip("already tested in testDetect")
	def testInit(self):
		testInstance = ai_sight.detection("C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\aisight\\obj_detect\\object_detection\\pretrained models\\rfcn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb",
										  "C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\aisight\\obj_detect\\object_detection\\labels\\mscoco_label_map.pbtxt"
										 )

		self.assertIsInstance(testInstance, ai_sight.detection)

	def testDetect(self):
		testInstance = ai_sight.detection("C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\aisight\\obj_detect\\object_detection\\pretrained models\\rfcn_resnet101_coco_11_06_2017\\frozen_inference_graph.pb",
										  "C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\aisight\\obj_detect\\object_detection\\labels\\mscoco_label_map.pbtxt"
										 )

		img = Image.open("C:\\Users\\Razin\\Documents\\Level 2\\Software Engineering group project\\project files\\aisight\\obj_detect\\object_detection\\test_images\\image1.jpg")
		np_image = load_image_into_numpy_array(img)

		(boxes, scores, classes, num) = testInstance.detect(np_image)

		

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':
	unittest.main()