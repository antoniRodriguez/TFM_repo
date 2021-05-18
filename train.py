import cv2
from matplotlib import pyplot as plt
import urllib.request
from tqdm import tqdm
import pandas as pd
import os
from build_dataset import create_csv_files


# EITHER LOAD WEIGHTS OR CREATE TRAIN/TEST CSV FILES
train = True
MODEL_NAME = '15_epochs_retinanet_pretrained_model.h5'



# TRAINING (RETINANET)
if train:
    PRETRAINED_MODEL = os.path.join('/home/group00/working/Antoni_THESIS/RETINANET','MODELS',str(MODEL_NAME))
    URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
    urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

    print("Downloaded pretrained model to " + PRETRAINED_MODEL)
else:
    # create dataset files
    base_path = "/home/group00/working/Antoni_THESIS/DATASETS/CCPD2019"
    create_csv_files(base_path)



'''
TRAIN
retinanet-train --freeze-backbone --batch-size 8 --steps 500 --weights '/home/group00/working/Antoni_THESIS/MODELS/15_epochs_retinanet_pretrained_model.h5' --epochs 15 csv ./train.csv ./classes.csv

CONVERT MODEL
retinanet-convert-model /home/group00/working/Antoni_THESIS/snapshots/resnet50_csv_100.h5 /home/group00/working/Antoni_THESIS/MODELS/INFERENCE_resnet50_csv_100.h5

TEST
retinanet-evaluate csv ./test.csv ./classes.csv /home/group00/working/Antoni_THESIS/MODELS/INFERENCE_resnet50_csv_100.h5


!keras_retinanet/bin/train.py 
--freeze-backbone 
--random-transform 
--weights {PRETRAINED_MODEL} 
--batch-size 8 
--steps 500 
--epochs 15 
csv annotations.csv classes.csv
'''