from django.test import TestCase
from .od_core.base64_numpy_conversion import base64_image_into_numpy_array
from .od_core import ai_sight

import numpy as np
from PIL import Image

class ConversionTest(TestCase):
    def test_Image(self):
        imgString = "/9j/4AAQSkZJRgABAQEAYABgAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAEAAAAAAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAAKABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD6y/a2/wCDk7w58GP2hvGvwt0fw3b6bq3hHVptHfVtfuJ7aKWWFjG22IwBMPIG8p/OZXQBv4iq9J/wT5/4KU/GL4+/GfQdLvND8b+OvD+tX8lrqGrt4GTRdD0ONUOWW+immEpXymPI2O0oUP8Ad2/oxqXhjTdbljkvdPsbySMfI88CyMv0JHFWtojVVUBVUYAHQVUpX0Wn9ef6ExjbfX+vI//Z"
        imgArr = np.array([[[250, 255, 251], [252, 255, 255], [252, 255, 255], [247, 250, 255], [252, 255, 255], [248, 251, 255],[251, 254, 255],[253, 254, 255],[252, 253, 255],[254, 255, 255],[253, 255, 252],[255, 255, 251],[250, 250, 248],[255, 255, 255],[255, 255, 255],[253, 252, 255]],[[252, 252, 252],[255, 255, 255],[255, 255, 255],[254, 254, 254],[250, 252, 251],[254, 255, 255],[250, 251, 253],[254, 255, 255],[252, 253, 255],[254, 255, 255],[252, 255, 255],[246, 249, 254],[252, 255, 255],[245, 249, 252],[250, 254, 255],[251, 255, 255]],[[255, 254, 255],[254, 252, 253],[253, 252, 250],[255, 255, 251],[255, 255, 250],[252, 253, 248],[244, 244, 242],[212, 213, 215],[162, 163, 168],[161, 164, 173],[ 99, 102, 111],[177, 180, 189],[170, 175, 181],[240, 245, 248],[251, 255, 255],[251, 255, 255]],[[252, 253, 255],[253, 254, 255],[254, 255, 255],[250, 252, 251],[254, 255, 255],[191, 193, 192],[155, 157, 156],[149, 150, 152],[ 81,  82,  86],[ 67,  68,  73],[ 77,  78,  83],[122, 123, 127],[131, 132, 134],[175, 177, 176],[247, 249, 246],[254, 255, 251]],[[247, 255, 255],[234, 245, 249],[ 81,  90,  97],[ 97, 106, 113],[ 89,  96, 106],[ 96, 100, 109],[105, 108, 115],[104, 105, 109],[168, 168, 168],[172, 172, 170],[215, 214, 209],[224, 221, 216],[181, 176, 170],[127, 122, 118],[145, 137, 134],[255, 250, 247]],[[243, 255, 255],[ 99, 112, 118],[ 99, 109, 118],[ 84,  94, 104],[ 78,  86,  99],[106, 112, 124],[126, 130, 139],[104, 105, 110],[ 77,  77,  77],[ 97,  96,  91],[106, 103,  94],[ 51,  47,  36],[ 67,  60,  50],[ 72,  63,  56],[187, 178, 173],[255, 252, 250]],[[249, 255, 255],[ 12,  19,  25],[  2,   9,  15],[ 23,  30,  36],[ 16,  21,  27],[ 51,  54,  59],[ 80,  84,  87],[102, 103, 105],[123, 123, 123],[ 50,  50,  48],[ 45,  44,  42],[ 69,  65,  62],[ 85,  81,  78],[ 75,  70,  66],[224, 219, 215],[255, 254, 250]],[[248, 247, 252],[123, 122, 127],[ 46,  46,  46],[ 21,  21,  19],[ 49,  50,  45],[ 26,  27,  22],[  8,   8,   6],[ 38,  40,  39],[ 86,  87,  89],[ 98,  99, 104],[167, 168, 173],[217, 218, 223],[244, 245, 249],[252, 253, 255],[254, 255, 253],[252, 254, 251]],[[255, 254, 255],[255, 254, 255],[250, 249, 247],[230, 229, 225],[209, 208, 203],[200, 201, 195],[207, 208, 203],[216, 216, 214],[242, 243, 245],[249, 250, 255],[254, 254, 255],[252, 255, 255],[251, 254, 255],[251, 255, 255],[251, 255, 255],[251, 255, 254]],[[254, 254, 254],[255, 255, 255],[255, 255, 255],[255, 255, 255],[254, 254, 255],[254, 254, 255],[254, 254, 255],[254, 254, 254],[251, 251, 251],[254, 254, 254],[255, 255, 255],[255, 255, 255],[253, 253, 253],[254, 254, 254],[254, 254, 255],[254, 254, 255]]])

        test_output = base64_image_into_numpy_array(imgString)
        
        self.assertTrue(np.array_equal(imgArr, test_output))
    
    def test_Image_Fail(self):
        incorrectString = "SGVsbG8="
        self.assertRaises(Exception, base64_image_into_numpy_array, incorrectString)

class Ai_SightTest(TestCase):

    def setUp(self):
        self.modelURL = "object_detection/od_core/pretrained_models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"
        self.labelURL = "object_detection/od_core/labels/mscoco_label_map.pbtxt"
        ai_sight.__init__(self.modelURL, self.labelURL)

    def test_init(self):
        img = Image.open("object_detection/test_image/test.jpg")
        (im_width, im_height) = img.size
        img_np = np.array(img).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

        boxes, scores, classes, _ = ai_sight.get_detection_result(img_np)

        boxestest = np.array([[0.20688292384147644, 0.02382349967956543, 0.8024313449859619, 0.9165056347846985]])
        self.assertTrue(np.array_equal(boxes, boxestest))
        
        scorestest = np.array([0.9100580811500549])
        self.assertTrue(np.array_equal(scores, scorestest))

        classestest = np.array([3])
        self.assertTrue(np.array_equal(classes, classestest))