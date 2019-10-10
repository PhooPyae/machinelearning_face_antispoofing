import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from pickle import dump
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, matrix_title):
    plt.figure(figsize=(9, 9), dpi=100)

    # use sklearn confusion matrix
    cm_array = confusion_matrix(y_true, y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))

    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

    plt.show()


if __name__ == '__main__':
#image must be cropped face image

    model = VGG16()
	#Modify model to remove the last layer
	model.layers.pop()
	model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
	print(model.summary())
	predictions = []
	prn = PRN(is_dlib = True)
	test_path = 'test_data'
	file = open('depth_features_labels/model_v1.pkl', 'rb')
	svm = pkl.load(file)

    test_path = 'test_data'
    model_loaded = load_model('models/model20191008-160032.h5')

    model_loaded.summary()
    
    is_live = 0
    is_spoof = 0

    live_threshold = 0.7

    X_test = []
    Y_test = []
    classes_name = []
    labels = 0
    target_size = (96,96)
    batch_size = 32
    spoof_score = 0
    for folder in os.listdir(test_path):
        if folder == '._DS_Store':
            continue
        folder_name = test_path+'/'+folder
        classes_name.append(folder)

        for files in os.listdir(folder_name):
            if files == '._DS_Store':
                continue
            file_name = folder_name+'/'+files
            print(file_name)
            image = cv2.imread(file_name)
            print('Image shape',np.shape(image))
            convface = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1,des1 = sift.detectAndCompute(convface,None)
            print('------------------')
            m,n,q = np.shape(image)
            for i in range(0,len(kp1)):
                p1 = int(kp1[i].pt[0])
                q1 = int(kp1[i].pt[1])
                if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
                    patch = image[p1-32:p1+32,q1-32:q1+32]
                    patch = cv2.resize(patch,target_size)
                    patch = np.reshape(patch,(1,patch.shape[0],patch.shape[1],patch.shape[2]))
                    prediction = model_loaded.predict(patch)
                    print(prediction)
                    spoof_score = spoof_score + prediction[1]
                    result = model_loaded.predict_classes(patch)
                    if result == 0:
                        live_count += 1
                    else :
                        spoof_count += 1
                    
                    if live_count > spoof_count:
                        print('live')
                        is_live += 1
                    else:
                        print('spoof')
                        is_spoof += 1

            image = cv2.resize(image,target_size)
			image_shape =  np.shape(image)
			print(np.shape(image))
			[h, w, c] = image_shape
			if c>3:
				image = image[:,:,:3]
			print(np.shape(image))
			pos = prn.process(image)
			vertices = prn.get_vertices(pos)
			depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
			print(vertices)
			imsave('depth_image.jpg',depth_image)
			load_image = load_img('depth_image.jpg',target_size=target_size)
			load_image = img_to_array(load_image)

			load_image = load_image.reshape((1, load_image.shape[0], load_image.shape[1], load_image.shape[2]))
			# # prepare the image for the VGG model
			load_image = preprocess_input(load_image)
			# get features
			img_feature = model.predict(load_image, verbose=0)
			predict = svm.predict(img_feature)
			print(predict)

            if is_live > is_spoof and predit[0] > live_threshold:
                print('LIVE')
            else:
                print('SPOOF')


            
