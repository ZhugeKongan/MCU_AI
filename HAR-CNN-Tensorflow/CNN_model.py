from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten
from keras import optimizers

def CNNmodel():

    model=Sequential()
    model.add(Conv2D(128, (2, 2), input_shape=(90, 3, 1), activation='relu'))
    # adding a maxpooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # adding a dropout layer for the regularization and avoiding over fitting
    model.add(Dropout(0.2))
    # flattening the output in order to apply the fully connected layer
    model.add(Flatten())
    # adding first fully connected layer with 256 outputs
    model.add(Dense(128, activation='relu'))
    # adding second fully connected layer 128 outputs
    model.add(Dense(128, activation='relu'))
    # adding softmax layer for the classification
    model.add(Dense(6, activation='softmax'))

    adam=optimizers.Adam(lr=0.001,decay=1e-6)
    model.compile(adam,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model=CNNmodel()
for layer in model.layers:
    print(layer.name)
