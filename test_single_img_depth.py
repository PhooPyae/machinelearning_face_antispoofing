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
	predictions = []
	prn = PRN(is_dlib = True)
	test_path = 'test_data'
	file = open('depth_features_labels/model.pkl', 'rb')
	model = pkl.load(file)

	X_test = []
	Y_test = []
	classes_name = []
	labels = 0
	target_size = (96,96)
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
			image = imread(file_name)
			[h, w, c] = image.shape
			if c>3:
				image = image[:,:,:3]
			print(np.shape(image))
			pos = prn.process(image)
			vertices = prn.get_vertices(pos)
			depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
			print(vertices)
			print(np.shape(depth_image))
			predict = model.predict(depth_image)
			print(predict)
			predictions.append(predict)

			tn, fp, fn, tp = confusion_matrix(list(Y_test),predictions).ravel()

			print("TP:",tp)
			print("TN:",tn)
			print("FP:",fp)
			print("FN:",fn)

			print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))