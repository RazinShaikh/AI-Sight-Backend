import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append("..")

from .utils import label_map_util

class detection:
    """docstring for detection"""
    def __init__(self, path_to_model, path_to_labels, num_classes = 90):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.num_classes = num_classes
        self.detection_graph = tf.Graph()
        
        with self.detection_graph.as_default():
            with tf.gfile.FastGFile(self.path_to_model, 'rb') as fid:
                od_graph_def = tf.GraphDef()
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes = self.num_classes, use_display_name = True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def detect(self, image_np):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})

                return boxes, scores, classes, num