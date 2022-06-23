
import numpy as np

from keras import optimizers
# from keras.applications.mobilenet_v2 import MobileNetV2
from model.mobilenet_kears import MobileNetv2
from model.model_test import cnnModel


random_seed = 611
np.random.seed(random_seed)
import os

class data_config:
    data_path = '/disks/disk2/lishengyan/MyProject/Fingerprint_Recognition_pytorch/dataset_FVC2000_DB4_B/dataset/'
    model_name = "mobilenet_baseline"
    MODEL_PATH = '/disks/disk2/lishengyan/MyProject/Fingerprint_Recognition_pytorch/ckpts/FVC2000/resnet50/mobilenet_baseline/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    WORKERS = 5
    epochs = 200#resnet:160,wresnet:200,resnext:300
    num_class = 10
    input_size = 64
    batch_size = 32
    delta =0.00001
    rand_seed=40
    lr=0.1
    warm=1#warm up training phase
    optimizer = "torch.optim.SGD"
    optimizer_parm = {'lr': lr,'momentum':0.9, 'weight_decay':5e-4, 'nesterov':False}
    # optimizer = "torch.optim.AdamW"
    # optimizer_parm = {'lr': 0.001, 'weight_decay': 0.00001}
    #scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    #scheduler_parm ={'milestones':[60,120,160], 'gamma':0.2}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max': 100, 'eta_min': 1e-4}
    loss_f ='torch.nn.CrossEntropyLoss'
    loss_dv = 'torch.nn.KLDivLoss'
    loss_fn = 'torch.nn.BCELoss'

# from source.prepare import Prepare_Data
from sklearn.model_selection import train_test_split
# # prepare data
# print('Prepare Dataset...')
# train_data_pre = Prepare_Data(data_config.input_size, data_config.input_size, data_config.data_path)
# img_data, label_data = train_data_pre.prepare_train_data()
# print('Finished: ', img_data.shape, label_data.shape)
#
# # split data
# print('Split Dataset for train and validation...')
# img_proc, img_test, label_proc, label_test = train_test_split(img_data, label_data, test_size = 0.1)
# # img_train, img_val, label_train, label_val = train_test_split(img_proc, label_proc, test_size = 0.1)
# print('Finished: ')
# trainX = img_proc
# testX = img_test
# trainY = label_proc
# testY = label_test

trainX = np.load(data_config.data_path+'img_train.npy')
testX = np.load(data_config.data_path+'img_test.npy')
valX = np.load(data_config.data_path+'img_val.npy')
trainY = np.load(data_config.data_path+'label_train.npy')
testY = np.load(data_config.data_path+'label_test.npy')
valY = np.load(data_config.data_path+'label_val.npy')
trainX=np.concatenate([trainX,valX],axis=0)
trainY=np.concatenate([trainY,valY],axis=0)
testX=np.concatenate([testX,valX],axis=0)
testY=np.concatenate([testY,valY],axis=0)
# trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size = 0.1,shuffle=True)

# img_data=np.load(data_config.data_path+'np_data/'+'img_train.npy')
# label_data=np.load(data_config.data_path+'np_data/'+'label_train.npy')
# img_proc, img_test, label_proc, label_test = train_test_split(img_data, label_data, test_size = 0.1,shuffle=True)
# trainX = img_data
# testX = img_test
# trainY = label_data
# testY = label_test

# trainX = np.load(data_config.data_path+'np_data/'+'img_train.npy')
# testX = np.load(data_config.data_path+'np_data/'+'img_real.npy')
# trainY = np.load(data_config.data_path+'np_data/'+'label_train.npy')
# testY = np.load(data_config.data_path+'np_data/'+'label_real.npy')
print(trainX.shape,testX.shape,trainY.shape,testY.shape)

# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5)
# model = MobileNetv2((64, 64, 1), 10)
model = cnnModel()
# model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

for layer in model.layers:
    print(layer.name)
model.fit(trainX,trainY, shuffle=True,validation_split=0,epochs=data_config.epochs,batch_size=data_config.batch_size,verbose=2)
score = model.evaluate(testX,testY,verbose=2)
print('test accuracy: %.2f%%' %(score[1]*100))
model.save('fpr_mobilenet.h5')
np.save('label_train.npy',trainY)
np.save('img_train.npy',trainX)

np.save('label_test.npy',testY)
np.save('img_test.npy',testX)


