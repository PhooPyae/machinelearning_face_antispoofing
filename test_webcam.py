import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from pickle import dump

model_loaded = load_model('models/model20190926-141500.h5')

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
# model = load_model('models/model_v3.h5')
target_size = (96,96)

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

# Read the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
color_red = (0,0,255)
line_width = 3
threshold = 1
frame_count = 0
while True:
    text = ''
    count = 0
    live_count = 0
    spoof_count = 0
    ret_val, frame = cam.read()
    frame =  cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    # image = cv2.imread(test_image)
    # print('Image shape',np.shape(image))
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    area = 0
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        # print('Confidence',confidence)
        if (confidence > 0.5):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            x,y,w,h = startX, startY, endX, endY
            x*=2
            y*=2
            w*=2
            h*=2
            if w*h > area:
                area = w*h
                print(area)
                if x < 0 or y < 0:
                    x = 0
                    y = 0
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                # cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                frame_count += 1
                face = frame[startY:endY, startX:endX]
                print('frame count',frame_count)
                if frame_count > 10:
                    frame_count = 0	
                    convface = cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
                    sift = cv2.xfeatures2d.SIFT_create()
                    kp1,des1 = sift.detectAndCompute(convface,None)
                    print('key point and destination',kp1,des1)
                    print('------------------')
                    m,n,q = np.shape(face)
                    for i in range(0,len(kp1)):
                        p1 = int(kp1[i].pt[0])
                        q1 = int(kp1[i].pt[1])
                        if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
                            patch = face[p1-32:p1+32,q1-32:q1+32]
                            print(np.shape(patch))
                            patch = cv2.resize(patch,target_size)
                            patch = np.reshape(patch,(1,patch.shape[0],patch.shape[1],patch.shape[2]))
                            print('patch shape',np.shape(patch))
                            prediction = model_loaded.predict(patch)
                            print(prediction)
                            result = model_loaded.predict_classes(patch)
                            if result == 0:
                                live_count += 1
                            else :
                                spoof_count += 1
                            
                            if live_count > spoof_count:
                                print('live')
                                text = 'Live'
                            else:
                                print('spoof')
                                text = 'spoof'
                            # dump(prediction, open("prediction"+".pkl", 'wb'))
                            # print('saved predicted model')
                            # # PREDICT CLASSES
                            # scores = model_loaded.evaluate(test_img, test_label, verbose=0)
                            # print(scores)
                            # print("%s: %.2f%%" % (model_loaded.metrics_names[1], scores[1]*100))
    # show the output frame

            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
    cv2.imshow("Frame", frame)
 
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) == 27:
        break  # esc to quit
 
# do a bit of cleanup
cv2.destroyAllWindows()
cam.release()

# prediction = model_loaded.predict(test_img)
# dump(prediction, open("prediction"+".pkl", 'wb'))
# print('saved predicted model')
# # PREDICT CLASSES
# scores = model_loaded.evaluate(test_img, test_label, verbose=0)
# print(scores)
# print("%s: %.2f%%" % (model_loaded.metrics_names[1], scores[1]*100))
