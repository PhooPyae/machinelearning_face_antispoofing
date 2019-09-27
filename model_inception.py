import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input
from keras.applications import xception

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

NUM_CLASSES = 2

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Download and create the pre-trained Xception model for transfer learning
base_model = xception.Xception(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have NUM_CLASSES classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Xception layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
optimizer = RMSprop(lr=0.001, rho=0.9)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=["accuracy"])
model.summary()

# TRAIN THE MODEL 
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

batch_size = 32
epochs = 100

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
classes = [0,1]
Y_pred = model.predict_generator(augmented_image.flow(X_test,Y_test), X_train.shape[0]//batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(classes, y_pred))
print('Classification Report')
target_names = ['Live', 'Spoof']
print(classification_report(classes, y_pred, target_names=target_names))
