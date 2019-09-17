from keras.models import load_model
from keras.metrics import accuracy_score

model = load_model('models/model.h5')
prediction = model.predict(X_test)
dump(prediction, open("prediction"+".pkl", 'wb'))
print('saved predicted model')
# PREDICT CLASSES
scores = model.evaluate(X_test, Y_test, verbose=0)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
