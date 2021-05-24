import cv2
from matplotlib import pyplot as plt
import urllib.request
from tqdm import tqdm
import pandas as pd
import os
import tensorflow_model_optimization as tfmot # (IMPORTANT)
import tensorflow as tf
#from build_dataset import create_csv_files


IMAGE_SIZE = (256,256,3)

# CUDA not properly set up yet
#print(tf.test.is_gpu_available())

# MODEL

input_ = tf.keras.layers.Input(shape = IMAGE_SIZE)

# CONV 1
c1 = tf.keras.layers.Conv2D(
    filters = 16, 
    kernel_size = (3, 3), 
    strides = (1, 1), 
    padding = 'SAME', 
    dilation_rate = (1, 1), 
    activation = None, 
    use_bias = True, 
    kernel_initializer = 'glorot_uniform', 
    bias_initializer = 'glorot_uniform')(input_)
c1BN = tf.keras.layers.BatchNormalization()(c1)
c1BN = tf.keras.layers.Activation('relu')(c1BN)
c1MP = tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), 
    strides=(2,2), 
    padding='SAME')(c1BN)

