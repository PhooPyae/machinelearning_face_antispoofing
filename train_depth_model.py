from sklearn import svm
import pickle as pkl
import numpy as np
import random
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,RandomizedSearchCV

def train_svm(feature_vector,labels):
    # parameters = {'C':[10, 100, 1000], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'poly', 'linear']}
    # svc = svm.SVC(kernel='rbf', # kernel type, rbf working fine here
    #                 degree=3, # default value, not tuned yet
    #                 coef0=1, # change to 1 from default value of 0.0
    #                 shrinking=True, # using shrinking heuristics
    #                 tol=0.01, # stopping criterion tolerance 
    #                 probability=False, # no need to enable probability estimates
    #                 cache_size=2000, # 200 MB cache size
    #                 class_weight=None, # all classes are treated equally 
    #                 verbose=False, # print the logs 
    #                 max_iter=-1, # no limit, let it run
    #                 decision_function_shape=None) # will use one vs rest explicitly 
                    
    # model = RandomizedSearchCV(svc, param_distributions=parameters, verbose=10, n_jobs=4)
    # model.fit(X_train, y_train)
    model = svm.SVC(kernel='poly',gamma=10,C=1000,probability=True)
    model.fit(feature_vector,labels)
    # print("best params: ", model.best_params_)
    return model

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

if __name__ == '__main__':

    predictions = []

    features_files = open('depth_features_labels/depth_features.pkl', 'rb')
    features = pkl.load(features_files)
    labels_files = open('depth_features_labels/depth_labels.pkl', 'rb')
    labels = pkl.load(labels_files)
    print(np.shape(features))
    print(np.shape(labels))

    nsamples, nx, ny = np.shape(features)
    features = np.reshape(features,(nsamples,nx*ny))
    
    print(np.shape(features))

    X_train, X_test, Y_train, Y_test = train_test_split(features,labels,test_size=0.2,shuffle=True)

    model = train_svm(X_train,Y_train)
    print(model)
    pkl.dump(model, open("depth_features_labels/model_v2"+".pkl", 'wb'))

    print("Now predicting...")
    for img in X_test:

        predictions.append(model.predict([img])[0])

    tn, fp, fn, tp = confusion_matrix(list(Y_test),predictions).ravel()

    print("TP:",tp)
    print("TN:",tn)
    print("FP:",fp)
    print("FN:",fn)

    print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))