import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append("..")

from .utils import label_map_util

detection_graph = None

def __init__(path_to_model, path_to_labels, num_classes = 90):
    global detection_graph
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        with tf.gfile.FastGFile(path_to_model, 'rb') as fid:
            od_graph_def = tf.GraphDef()
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = num_classes, use_display_name = True)
    category_index = label_map_util.create_category_index(categories)

def detect(image_np):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                    feed_dict={image_tensor: image_np_expanded})

            return boxes, scores, classes, num
