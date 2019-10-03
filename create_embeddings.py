from os import listdir
from pickle import dump
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

#
def extract_features(directory,target_size):
    labels = []
    features = []
    model = VGG16()
    #Modify model to remove the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
    print(model.summary())

    for sub_dir in listdir(directory):
        if sub_dir == '.DS_Store':
            continue

        if sub_dir == 'live':
            subdirectory = directory + "/"+ sub_dir

            for img_name in listdir(subdirectory):
                filename = subdirectory+ "/" + img_name
                print(filename)
                if img_name == 'Thumbs.db' or img_name == '.DS_Store':
                    continue

                image = load_img(filename,target_size=(target_size,target_size))
                image = img_to_array(image)

                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                # get features
                img_feature = model.predict(image, verbose=0)
                # store feature
                # features[subsub+"\\"+img_name] = img_feature
                # print('>%s' % filename)
                features.append(img_feature)
                labels.append(0)
            # dump(features, open("features/live_depth_features" + ".pkl", 'wb'))


        if sub_dir == 'spoof':
            subdirectory = directory + "/" + sub_dir

            for img_name in listdir(subdirectory):
                filename = subdirectory+ "/" + img_name
                print(filename)
                if img_name == 'Thumbs.db' or img_name == '.DS_Store':
                    continue

                image = load_img(filename, target_size=(target_size, target_size))
                image = img_to_array(image)

                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                # get features
                img_feature = model.predict(image, verbose=0)
                # store feature
                # features[subsub + "\\" + img_name] = img_feature
                # print('>%s' % filename)
                features.append(img_feature)
                labels.append(1)
    dump(features, open("depth_features_labels/depth_features"+".pkl", 'wb'))
    dump(labels,open("depth_features_labels/depth_labels"+".pkl",'wb'))



img_features = extract_features("depth_data",224)


