from CNN_model import  CNNmodel
from util import *
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

filePath='/disks/disk2/lishengyan/mcuev/HAR-CNN-TensorFlow/actitracker_raw.txt'
print('*****load data and processing******')
dataset=readdata(filePath)
segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels),dtype = np.int8)

# number of epochs
Epochs = 100
# batchsize
batchSize = 64
trainSplitRatio = 0.8
numOfRows = segments.shape[1]#(24403, 90, 3)
numOfColumns = segments.shape[2]
reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
# splitting in training and testing data
# print(reshapedSegments.shape)#(24403, 90, 3)

# trainSplit =int(len(reshapedSegments)*trainSplitRatio)# np.random.rand(len(reshapedSegments)) < trainSplitRatio
# trainX = reshapedSegments[:trainSplit]
# testX = reshapedSegments[trainSplit:]
# trainX = np.nan_to_num(trainX)#使用0代替数组x中的nan元素,使用有限的数字代替inf元素
# testX = np.nan_to_num(testX)
# trainY = labels[:trainSplit]
# testY = labels[trainSplit:]
trainSplit =np.random.rand(len(reshapedSegments)) < trainSplitRatio
trainX = reshapedSegments[trainSplit]
testX = reshapedSegments[~trainSplit]
trainX = np.nan_to_num(trainX)#使用0代替数组x中的nan元素,使用有限的数字代替inf元素
testX = np.nan_to_num(testX)
trainY = labels[trainSplit]
testY = labels[~trainSplit]
print("trainSplit:",len(trainX),len(testX))

print('*****load model******')
model=CNNmodel()
for layer in model.layers:
    print(layer.name)

print('*****traing******')
model.fit(trainX,trainY, validation_split=1-trainSplitRatio,epochs=Epochs,batch_size=batchSize,verbose=2)
score = model.evaluate(testX,testY,verbose=2)

print('Accuracy: %.2f%%' %(score[1]*100))
predictions = model.predict(testX)
model.save('model.h5')

np.save('groundTruth.npy',testY)
np.save('testData.npy',testX)


