import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
from pickle import dump
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.multiclass import unique_labels

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

    test_path = 'test_data'
    model_loaded = load_model('models/model20190926-141500.h5')

    model_loaded.summary()


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
            print(file_name)
            image = cv2.imread(file_name)
            print('Image shape',np.shape(image))
            convface = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1,des1 = sift.detectAndCompute(convface,None)
            print('------------------')
            m,n,q = np.shape(image)
            for i in range(0,len(kp1)):
                p1 = int(kp1[i].pt[0])
                q1 = int(kp1[i].pt[1])
                if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
                    patch = image[p1-32:p1+32,q1-32:q1+32]
                    patch = cv2.resize(patch,target_size)
                    X_test.append(patch)
                    Y_test.append(labels)
        labels += 1


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
        
    print(np.shape(X_test))
    print(np.shape(Y_test))
    Y_test = to_categorical(Y_test,2)
    
    print(Y_test)
    probabilities = model_loaded.predict(X_test)
    probabilities = np.round(probabilities)
    print(probabilities)
    plot_confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(probabilities,axis=1),'Confusion Matrix')
    cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(probabilities,axis=1))
    print('Confusion Matrix')
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(tn, fp, fn, tp)
    # plot_confusion_matrix(Y_test,probabilities,"Confusion matrix")


    # # Y_pred = model.predict_generator(test_gen.flow(X_test,Y_test), X_test.shape[0] // batch_size+1)
    # # y_pred = np.argmax(Y_pred, axis=1)
    # # print('Confusion Matrix')
    # # print(confusion_matrix(validation_generator.classes, y_pred))
    # # print('Classification Report')
    # # target_names = ['Cats', 'Dogs', 'Horse']
    # # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


    # prediction = model_loaded.predict(X_test)
    # dump(prediction, open("prediction"+".pkl", 'wb'))
    # print('saved predicted model')
    # # PREDICT CLASSES
    # scores = model_loaded.evaluate(X_test, Y_test, verbose=0)
    # print(scores)
    # print("%s: %.2f%%" % (model_loaded.metrics_names[1], scores[1]*100))
