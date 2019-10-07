from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import models
from keras import layers
from keras.models import Model, Sequential

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = {224,224}

#train_path = 'training_set'
#valid_path = 'training_set'



vgg = VGG16(input_shape=(224, 224,3), weights=None, include_top=False) # weights='imagenet'

for layer in vgg.layers:
	layer.trainable  = False



#folders = glob('training_set/*')
num_classes = 2
x = Flatten()(vgg.output)

#prediction = Dense(len(folders), activation = 'softmax')(x)
prediction = Dense(num_classes, activation= 'softmax')(x)

model = Model(inputs=vgg.input, output=prediction)

model.summary()
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])



epochs = 50
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	'depth_data/train',
    target_size = (224,224),
    batch_size=32)


validation_generator = val_datagen.flow_from_directory(
	'depth_data/valid',
    target_size = (224,224),
    batch_size=10)

history=model.fit_generator(train_generator,
                         steps_per_epoch = 30,
                         epochs = epochs,
                         validation_data = validation_generator,
                         validation_steps = 20)

import numpy as np
# compute predictions
predictions = model.predict_generator(generator=validation_generator,steps=10)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()
print(y_test,'classes')
print(class_names,'class name')


import matplotlib.pyplot as plt



plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
plt.savefig('model_output/vgg_acc.jpg')


#plt.figure(figsize=(20,10))
#plt.subplot(1, 2, 1)
#plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
plt.savefig('model_output/vgg_loss.jpg')

#plt.subplot(1, 2, 2)
#plt.ylabel('Accuracy', fontsize=16)
#plt.figure(figsize=(20,10))


import tensorflow as tf
from keras.models import load_model

model.save('model_output/vgg16_model1.h5')

















