import pickle
import keras
import tflearn
from cnn import cnn

from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split

from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential,Model

with open('imgs.pickle', 'rb') as f:
    imgs = pickle.load(f)

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)

x, x_test, y, y_test = train_test_split(imgs,labels,test_size=0.2, shuffle = True)
y = tflearn.data_utils.to_categorical(y,2)
y_test = tflearn.data_utils.to_categorical(y_test,2)

print(x.shape)
print(x_test.shape)
print(y.shape)
print(y_test.shape)

# logdir="logs/scalars/"


model = tflearn.DNN (cnn(), clip_gradients=5.0, tensorboard_verbose=2, tensorboard_dir='/tmp/tflearn_logs/', 
            checkpoint_path='/tmp/checkpoint_path/', best_checkpoint_path='/tmp/best_checkpoint_path/', max_checkpoints=None, session=None, best_val_accuracy=0.0)
# model = tflearn.DNN(cnn(),checkpoint_path = 'model_DNN',max_checkpoints=10,
# tensorboard_verbose = 2)
# model = tflearn.DNN(cnn(),tensorboard_verbose = 3,tensorboard_dir='/tmp/tflearn_logs/')
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],learning_rate=0.001)
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(x,y,n_epoch = 10, validation_set = (x_test,y_test), batch_size=500, shuffle = True)
model.save('cnn_model_edit.model')
# model = Sequential([
#     Conv2D(filters=32,input_shape=(96,96,3),kernel_size=(3,3),activation='relu'),
#     MaxPool2D(pool_size=(2,2)),

#     Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
    
#     Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
    
#     Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
    
#     Conv2D(filters=256,kernel_size=(3,3),activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
    
#     Flatten(),
    
#     Dense(256),
#     Dense(2,activation='softmax')
# ])
# model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
# model.fit(x,y,epochs=10,validation_data=(x_test,y_test),batch_size=2)

# model.save('patch_based.h5')