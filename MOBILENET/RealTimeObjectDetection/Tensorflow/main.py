




######
WORKSPACE_PATH        = 'Tensorflow/workspace'
SCRIPTS_PATH          = 'Tensorflow/scripts'
APIMODEL_PATH         = WORKSPACE_PATH + '/models'
ANNOTATION_PATH       = WORKSPACE_PATH + '/annotations'
IMAGE_PATH            = WORKSPACE_PATH + '/images'
MODEL_PATH            = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH           = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH       = MODEL_PATH + '/my_ssd_mobnet/'



#python scripts/generate_tfrecord.py -x 'workspace/images/train' -l 'workspace/annotations/label_map.pbtxt' -o 'workspace/annotations/train.record'