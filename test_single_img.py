import argparse
import numpy as np
from keras.models import load_model
import cv2
from sklearn.metrics import accuracy_score
import os
from pickle import dump
from keras.utils import to_categorical
import sklearn

def main(args):
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
            patch = np.reshape(patch,(1,patch.shape[0],patch.shape[1],patch.shape[2]))
            # print(np.shape(patch))
            prediction = model_loaded.predict_proba(patch)
            print(prediction)
            predict_class = model_loaded.predict_classes(patch)
            print('predict classes',predict_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test result with single face image')
    parser.add_argument('-i', '--inputImage', default='test_image.jpg', type=str,
                        help='path to the input directory, where input images are stored.')
                        
    main(parser.parse_args())