import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

import pickle
from pickle import dump

import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datetime import datetime

train_path = 'dataset'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(96,96),color_mode='rgb',class_mode='categorical',batch_size=800,shuffle=True)
X, y = next(train_batches)
print(np.shape(X))
print(np.shape(y))

print('features and labels are loaded')

# close the file
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2)

# X_train = np.array(X_train)
# X_test = np.array(X_test)

# Y_train = np.array(Y_train) 
# Y_test = np.array(Y_test)

# nb_classes = 2
# Y_train = to_categorical(Y_train, nb_classes)
# Y_test = to_categorical(Y_test, nb_classes)

# input_shape = (64, 64, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))

augmented_image = ImageDataGenerator(
    shear_range=0.1,
)
# augmented_image.fit(X_train)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#BUILD THE MODEL
model = Sequential()

model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=(96,96,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(16, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(16, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(16, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(32, (3, 3),strides=(1,1), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(3000, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(400, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(2,activation='softmax'))
model.summary()

# TRAIN THE MODEL 
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

batch_size = 16
epochs = 150

model.fit_generator(augmented_image.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_test,Y_test),
verbose=1,steps_per_epoch = X_train.shape[0]//batch_size,callbacks=[tensorboard_callback],)
# # model.fit(X_train, Y_train, batch_size=32, epochs=20,
# #                  verbose=1, validation_data=(X_test, Y_test))

#save model
model.save('models/model'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.h5')
print("Saved model to disk")

prediction = model.predict(X_test)
dump(prediction, open("prediction"+".pkl", 'wb'))
print('saved predicted model')
# PREDICT CLASSES
scores = model.evaluate(X_test, Y_test, verbose=0)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(augmented_image.flow(X_test,Y_test), X_train.shape[0]//batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(Y_test, y_pred))
print('Classification Report')
target_names = ['Live', 'Spoof']
print(classification_report(classes, y_pred, target_names=target_names))
