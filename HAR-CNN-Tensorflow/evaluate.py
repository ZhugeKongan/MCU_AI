from keras.models import load_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os

model=load_model('model.h5')
test_x=np.load('testData.npy')
groundTruth = np.load('groundTruth.npy')

for layer in model.layers:
    print(layer.name)

score = model.evaluate(test_x,groundTruth,verbose=2)
print('Baseline Error: %.2f%%' %(100-score[1]*100))

labels = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']

predictions = model.predict(test_x,verbose=2)
predictedClass = np.zeros((predictions.shape[0]))
for instance in range (predictions.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance,:])
    print(labels[int(predictedClass[instance])])

