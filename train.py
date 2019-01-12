from __future__ import division, absolute_import
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys
import tensorflow as tf
import os
import glob
import cv2
import sys


# prevents appearance of tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

target_classes = ['angry', 'disgusted',
                  'fearful', 'happy', 'sad', 'surprised', 'neutral']


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in target_classes:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)  # open image
            newimg = format_image(image)
            training_data.append(newimg)
            training_labels.append(target_classes.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # open image
            newimg = format_image(image)
            prediction_data.append(newimg)
            prediction_labels.append(target_classes.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def get_files(emotion):
    print("./images/%s/*" % emotion)
    files = glob.glob("./images/%s/*" % emotion)
    print(len(files))

    random.shuffle(files)
    # get first 80% of file list
    training = files[:int(len(files)*0.8)]
    # get last 20% of file list
    prediction = files[-int(len(files)*0.2):]

    return training, prediction


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
            # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(
            image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image


network = input_data(shape=[None, 48, 48, 1])
print("Input data     ", network.shape[1:])

network = conv_2d(network, 64, 5, activation='relu')
print("Conv1          ", network.shape[1:])

network = max_pool_2d(network, 3, strides=2)
print("Maxpool1       ", network.shape[1:])

network = conv_2d(network, 64, 5, activation='relu')
print("Conv2          ", network.shape[1:])

network = max_pool_2d(network, 3, strides=2)
print("Maxpool2       ", network.shape[1:])

network = conv_2d(network, 128, 4, activation='relu')
print("Conv3          ", network.shape[1:])

network = dropout(network, 0.3)
print("Dropout        ", network.shape[1:])

network = fully_connected(network, 3072, activation='relu')
print("Fully connected", network.shape[1:])

network = fully_connected(network, len(target_classes), activation='softmax')
print("Output         ", network.shape[1:])
# Generates a TrainOp which contains the information about optimization process - optimizer, loss function, etc

network = regression(network, optimizer='momentum',
                     metric='accuracy', loss='categorical_crossentropy')

# Creates a model instance.
model = tflearn.DNN(network, checkpoint_path='model_1_atul',
                    max_checkpoints=1, tensorboard_verbose=2)


training_data, training_labels, prediction_data, prediction_labels = make_sets()

# model.fit(training_data, training_labels, n_epoch=10, validation_set=None)
# model.save("model_test_atul.tflearn")
