import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from pickle import dump
from keras.utils import to_categorical

#image must be cropped face image

test_path = 'test_data'
model_loaded = load_model('models/model20190926-141500.h5')

model_loaded.summary()

# base_dir = os.path.dirname(__file__)
# prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
# caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# # Read the model
# model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


X_test = []
Y_test = []
labels = 0
target_size = (96,96)
for folder in os.listdir(test_path):
    if folder == '._DS_Store':
        continue
    folder_name = test_path+'/'+folder

    for files in os.listdir(folder_name):
        if files == '._DS_Store':
            continue
        file_name = folder_name+'/'+files
        print(file_name)
        image = cv2.imread(file_name)
        print('Image shape',np.shape(image))
        # (h, w) = image.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # model.setInput(blob)
        # detections = model.forward()
        # for i in range(0, detections.shape[1]):
        #     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #     (startX, startY, endX, endY) = box.astype("int")

        #     confidence = detections[0, 0, i, 2]

        #     # If confidence > 0.5, show box around face
        #     print('Confidence',confidence)
        #     if (confidence > 0.5):
        #         # cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
        #         subface = image[startY:endY , startX:endX]
        #         print(np.shape(subface))
        convface = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1,des1 = sift.detectAndCompute(convface,None)
        print('key point and destination',kp1,des1)
        print('------------------')
        m,n,q = np.shape(image)
        for i in range(0,len(kp1)):
            p1 = int(kp1[i].pt[0])
            q1 = int(kp1[i].pt[1])
            if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
                patch = image[p1-32:p1+32,q1-32:q1+32]
                patch = cv2.resize(patch,target_size)
                X_test.append(patch)
                Y_test.append(labels)
    labels += 1

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test,2)
X_test = X_test.astype('float32')
X_test /= 255


prediction = model_loaded.predict(X_test)
dump(prediction, open("prediction"+".pkl", 'wb'))
print('saved predicted model')
# PREDICT CLASSES
scores = model_loaded.evaluate(X_test, Y_test, verbose=0)
print(scores)
print("%s: %.2f%%" % (model_loaded.metrics_names[1], scores[1]*100))
