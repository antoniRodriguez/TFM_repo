import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import time
import pickle
from xml.dom import minidom
import xml.etree.cElementTree as ET

PROJECT_PATH          = 'MOBILENET/RealTimeObjectDetection'
WORKSPACE_PATH        = PROJECT_PATH + '/Tensorflow/workspace'
SCRIPTS_PATH          = PROJECT_PATH + '/Tensorflow/scripts'
APIMODEL_PATH         = WORKSPACE_PATH + '/models'
ANNOTATION_PATH       = WORKSPACE_PATH + '/annotations'
IMAGE_PATH            = WORKSPACE_PATH + '/images'
MODEL_PATH            = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH           = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH       = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME     = 'my_ssd_mobnet'



# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

detections_list = []
total_elapsed_time = 0
for i in tqdm(range(1000)):
    filename = str(i).zfill(7)+'.jpg'

    image = cv2.cvtColor(cv2.imread('/home/group00/working/Antoni_THESIS/MOBILENET/RealTimeObjectDetection/Tensorflow/workspace/images/val/'+filename), cv2.COLOR_BGR2RGB)
    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    start = time.time()
    detections = detect_fn(input_tensor)
    end = time.time()
    if i > 0:
        total_elapsed_time += (end-start)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.15,
                agnostic_mode=False)
    detections_list.append(detections)

    cv2.imwrite(os.path.join(PROJECT_PATH,'output_images',filename),  cv2.resize(image_np_with_detections, (800, 600)))

print("MEAN ELAPSED TIME PER DETECTION: " + str(total_elapsed_time/1000) + 's')


with open(os.path.join(PROJECT_PATH,'output_images','detections.pkl'), 'wb') as f:
    pickle.dump(detections_list, f)