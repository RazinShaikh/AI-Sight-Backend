import numpy as np
import sys
import tensorflow as tf

sys.path.append("..")

from .utils import label_map_util, visualization_utils as vis_util  # noqa

detection_graph = None
category_index = None


def __init__(path_to_model, path_to_labels, num_classes=90):
    global detection_graph
    global category_index

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        with tf.gfile.FastGFile(path_to_model, 'rb') as fid:
            od_graph_def = tf.GraphDef()
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


def detect(image_np):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name(
                'image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes,
                 detection_scores,
                 detection_classes,
                 num_detections],
                feed_dict={image_tensor: image_np_expanded})

            return boxes, scores, classes, num


def get_detection_result(image_np, max_boxes=20, min_score_thresh=0.5):

    boxes_res = []
    scores_res = []
    classes_res = []
    display_string = []

    boxes, scores, classes, _ = detect(image_np)

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)

    for i in range(min(max_boxes, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            boxes_res.append(boxes[i])
            scores_res.append(scores[i])
            classes_res.append(classes[i])

            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'

            display_str = '{}: {}%'.format(
                class_name,
                int(100*scores[i]))
            display_string.append(display_str)

    return boxes_res, scores_res, classes_res, display_string


def draw_objects_boxes(image_np):
    boxes, scores, classes, _ = detect(image_np)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=30
    )

    return image_np
