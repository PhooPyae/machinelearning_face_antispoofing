import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from pickle import dump

model_loaded = load_model('models/model.h5')

model_loaded.summary()

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


test_img = []
test_label = []
labels = 0

#load model
scale_factor = 1.2
min_neighbors = 1
min_size = (50, 50)
target_size = 64
model = load_model('models/model_v3.h5')
target_size = 64

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
color_red = (0,0,255)
line_width = 3
threshold = 1
while True:
    count = 0
    live_count = 0
    spoof_count = 0
    ret_val, img = cam.read()
    frame_small =  cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = frame_small[:,:,::-1]

    # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

prediction = model_loaded.predict(test_img)
dump(prediction, open("prediction"+".pkl", 'wb'))
print('saved predicted model')
# PREDICT CLASSES
scores = model_loaded.evaluate(test_img, test_label, verbose=0)
print(scores)
print("%s: %.2f%%" % (model_loaded.metrics_names[1], scores[1]*100))
