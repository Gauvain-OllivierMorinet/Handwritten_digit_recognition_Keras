import numpy as np
import preparedata as pr
import cnnutils as cu
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('th')

# fix random seed for reproductibility
seed = 7
np.random.seed(seed)


# define the medium model
def medium_model():
    #create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# import prepared MNIST Dataset

(X_train, Y_train), (X_test, Y_test), num_classes = pr.get_and_prepare_data_mnist()

# building model
model = medium_model()

#training model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200)

# Evaluate the model
cu.print_model_error_rate(model, X_test, Y_test)