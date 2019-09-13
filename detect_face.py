import keras
import cv2
import numpy as np
import os
import tflearn
import matplotlib
import pickle

from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split

from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential,Model

# def detect(img, cascade):
#     rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(10, 10),
#                                      flags=cv2.CASCADE_SCALE_IMAGE)
#     if len(rects) == 0:
#         return []
#     rects[:,2:] += rects[:,:2]
#     return rects


# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if __name__ == '__main__':
    # Define paths
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
    caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

    # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    labels = []
    data=[]
    count = 0

    #LOAD IMAGES
    train_path = 'face_to_crop'
    for folder in os.listdir(train_path):
        if(folder == '.DS_Store'):
            continue
        folder_name = train_path+'/'+folder
        for subfolder in os.listdir(folder_name):

            if(subfolder == '.DS_Store' or subfolder == 'Thumbs.db'):
                continue
            subfolder_name = folder_name+'/'+subfolder

            for files in os.listdir(subfolder_name):
                if(files == '.DS_Store' or files == 'Thumbs.db'):
                    continue
                file_name = subfolder_name+'/'+files
                print('file name',file_name)
                image = cv2.imread(file_name)
                print('Image shape',np.shape(image))
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                model.setInput(blob)
                detections = model.forward()
                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    confidence = detections[0, 0, i, 2]

                    # If confidence > 0.5, show box around face
                    if (confidence > 0.5):
                        # cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
                        subface = image[startY:endY , startX:endX]
                        print('Face shape',np.shape(subface))
                        cv2.imwrite('updated_images/'+folder+'/'+ files, subface)
                        print("Image " + file_name + " converted successfully")