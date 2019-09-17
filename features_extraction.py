from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pickle import dump
import numpy as np
import os

#PREPROCESSING
DatasetPath = []
data_path = 'dataset'
imageData = []
imageLabels = []
label_count = 0

for folder in os.listdir(data_path):
    if(folder == '.DS_Store'):
        continue
    folder_name = data_path+'/'+folder
    for sub_folder in os.listdir(folder_name):
        if(sub_folder == '.DS_Store'):
            continue
        sub_folder_name = folder_name+'/'+sub_folder
        for files in os.listdir(sub_folder_name):
            if(files == '.DS_Store'):
                continue
            file_name = sub_folder_name+'/'+files
            print(file_name)
            imgRead = load_img(file_name,target_size = (96,96))
            imgRead = img_to_array(imgRead)
            imageData.append(imgRead)
            imageLabels.append(label_count)
    label_count += 1
    print(imageLabels)
    print(np.shape(imageData))

    dump(imageData, open("train_features"+".pkl", 'wb'))
    print('features are saved')
    dump(imageLabels, open("train_labels"+".pkl", 'wb'))
    print('labels are saved')