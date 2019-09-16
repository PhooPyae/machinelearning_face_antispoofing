from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

from pickle import dump

import numpy as np
import os

from sklearn.model_selection import train_test_split

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
            imgRead = load_img(file_name,target_size = (96,96))
            imgRead = img_to_array(imgRead)
            imageData.append(imgRead)
            imageLabels.append(label_count)
        label_count += 1
        print(imageLabels)

X_train, X_test, Y_train, Y_test = train_test_split(imageData,imageLabels,test_size=0.2)

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train) 
Y_test = np.array(Y_test)

nb_classes = 3
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

# input_shape = (64, 64, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# BUILD THE MODEL
model = Sequential()
model.add(Conv2D(50, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=(96,96,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Conv2D(150, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Conv2D(200, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Conv2D(250, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2,activation='softmax'))
model.summary()

# TRAIN THE MODEL 
adam = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


model.fit(X_train, Y_train, batch_size=5, epochs=30,
                 verbose=1, validation_data=(X_test, Y_test))

#save model
model.save('models/model.h5')
print("Saved model to disk")

prediction = model.predict(X_test)
dump(prediction, open("prediction"+".pkl", 'wb'))
print('saved predicted model')
# PREDICT CLASSES
scores = model.evaluate(X_test, Y_test, verbose=0)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

