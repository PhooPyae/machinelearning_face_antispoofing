import numpy as np
from keras.models import load_model
import cv2
from sklearn.metrics import accuracy_score
from pickle import dump
from keras.utils import to_categorical
import sklearn
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import ast
from api import PRN

if __name__='__main__':
    prn = PRN(is_dlib = True)
    test_image = args.inputImage
    target_size = (96,96)
    
    #image must be cropped face image
    model_loaded = load_model('models/model20190926-141500.h5')

    model_loaded.summary()
    
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
    caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

    # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    image = cv2.imread(test_image)
    # print('Image shape',np.shape(image))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[1]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        print('Confidence',confidence)
        if (confidence > 0.5):
            # cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
            subface = image[startY:endY , startX:endX]
            print(np.shape(subface))
            pos = prn.process(subface)
            vertices = prn.get_vertices(pos)
            depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
            print(vertices)
            print(np.shape(vertices))