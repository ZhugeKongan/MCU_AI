from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout,Dense,GlobalMaxPooling2D,MaxPooling2D,Flatten
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.layers import DepthwiseConv2D
from keras import backend as K
from keras import optimizers

def cnnModel():
    model = Sequential()
    # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
    model.add(Conv2D(32, (5,5),input_shape=(64, 64,1),activation='relu',strides=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu',strides=(2,2)))
    # adding a maxpooling layer
    model.add(GlobalMaxPooling2D())
    # adding a dropout layer for the regularization and avoiding over fitting
    model.add(Dropout(0.3))
    # flattening the output in order to apply the fully connected layer
    # model.add(Flatten())
    # adding first fully connected layer with 256 outputs
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    #adding second fully connected layer 128 outputs
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    # adding softmax layer for the classification
    model.add(Dense(10, activation='softmax'))
    # Compiling the model to generate a model
    adam = optimizers.Adam(lr = 0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

if __name__ == '__main__':
     model=cnnModel()
     model.save('model1.h5')