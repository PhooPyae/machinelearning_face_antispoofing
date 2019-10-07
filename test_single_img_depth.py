import numpy as np
from keras.models import load_model
import cv2
from sklearn.metrics import accuracy_score
import pickle as pkl
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
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.metrics import confusion_matrix

def predict_image(svm_model,path,target_size=224):
	model = VGG16()
	# Modify model to remove the last layer
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	image = load_img(path, target_size=(target_size, target_size))
	image = img_to_array(image)

	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	img_feature = model.predict(image, verbose=0)
	return svm_model.predict(list(img_feature))

if __name__=='__main__':
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

	labels = []
	classes_name = []
	label = 0
	target_size = (224,224)
	batch_size = 32
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
			image = imread(file_name)
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
			predictions.append(predict)
			labels.append(label)
		label =+ 1

		tn, fp, fn, tp = confusion_matrix(list(labels),predictions).ravel()

		print("TP:",tp)
		print("TN:",tn)
		print("FP:",fp)
		print("FN:",fn)

		print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))