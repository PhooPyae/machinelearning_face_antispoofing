import numpy as np
import cv2
import sklearn
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.multiclass import unique_labels

import pickle as pkl
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import ast
from api import PRN
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

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
	test_path = 'sample_test_data'
	file = open('depth_features_labels/model_v2.pkl', 'rb')
	svm = pkl.load(file)
	model_loaded = load_model('models/good/model20191008-160032_73.h5')

	model_loaded.summary()
	
	is_live = 0
	is_spoof = 0

	spoof_threshold = 0.2
	live_threshold = 0.7

	X_test = []
	Y_test = []
	classes_name = []
	final_result = []
	labels = 0
	target_size = (96,96)
	batch_size = 32
	spoof_score = 0
	folder_count = 0
	for folder in os.listdir(test_path):
		if folder == '._DS_Store':
			continue
		folder_name = test_path+'/'+folder
		classes_name.append(folder_count)

		for files in os.listdir(folder_name):
			if files == '._DS_Store':
				continue
			live_count = 0
			spoof_count = 0
			file_name = folder_name+'/'+files
			print(file_name)
			image = cv2.imread(file_name)
			# print('Image shape',np.shape(image))
			# convface = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
			sift = cv2.xfeatures2d.SIFT_create()
			kp1,des1 = sift.detectAndCompute(image,None)
			print('------------------')
			count = 0
			m,n,q = np.shape(image)
			for i in range(0,len(kp1)):
				p1 = int(kp1[i].pt[0])
				q1 = int(kp1[i].pt[1])
				if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
					patch = image[p1-32:p1+32,q1-32:q1+32]
					count += 1
					patch_name = str(count)+'.jpg'
					cv2.imwrite('patches/'+patch_name,patch)
					patch = cv2.resize(patch,target_size)
					patch = np.reshape(patch,(1,patch.shape[0],patch.shape[1],patch.shape[2]))
					prediction = model_loaded.predict(patch)
					# print(prediction)
					result = model_loaded.predict_classes(patch)
					# print('classes',result)
					if result == 0:
						live_count += 1
					else :
						spoof_count += 1
					
					# if live_count > spoof_count:
					# 	is_live += 1
					# else:
					# 	is_spoof += 1
				
			print('live_count',live_count)
			print('spoof_count',spoof_count)

			total_count = live_count+spoof_count

			live_probability = live_count/ total_count
			spoof_probability = spoof_count/total_count

			live_probability = round(live_probability,1)
			spoof_probability = round(spoof_probability,1)

			print('live_probability',live_probability)
			print('spoof_probability',spoof_probability)

			image = cv2.resize(image,(224,224))
			image_shape =  np.shape(image)
			print(np.shape(image))
			[h, w, c] = image_shape
			if c>3:
				image = image[:,:,:3]
			print(np.shape(image))
			pos = prn.process(image)
			vertices = prn.get_vertices(pos)
			depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
			# print(vertices)
			imsave('depth_image.jpg',depth_image)
			load_image = load_img('depth_image.jpg',target_size=(224,224))
			load_image = img_to_array(load_image)

			load_image = load_image.reshape((1, load_image.shape[0], load_image.shape[1], load_image.shape[2]))
			# # prepare the image for the VGG model
			load_image = preprocess_input(load_image)
			# get features
			vgg_prediction = model.predict(load_image, verbose=0)
			print('Prediction VGG',vgg_prediction)
			predict = svm.predict(vgg_prediction)
			print(predict)
			probability = svm.predict_proba(vgg_prediction)
			print('probability', probability)
			
			for prob in probability:
				print(prob)
				if (live_probability > spoof_threshold and spoof_probability < live_threshold) and (prob[0] > spoof_threshold and prob[1] < live_threshold):
					print('LIVE')
					final_result.append(0)
				else:
					print('SPOOF')
					final_result.append(1)
		folder_count += 1
	print(classes_name)
	print(final_result)
	pkl.dump(classes_name, open("true_label"+".pkl", 'wb'))
	pkl.dump(final_result, open("predicted_label"+".pkl", 'wb'))
		



			
