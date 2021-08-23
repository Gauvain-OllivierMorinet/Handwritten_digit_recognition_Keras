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


# define the small CNN model
def small_model():
    #create model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # Model compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# import prepared MNIST Dataset

(X_train, Y_train), (X_test, Y_test), num_classes = pr.get_and_prepare_data_mnist()

# building model
model = small_model()

#training model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200)

# Evaluate the model
cu.print_model_error_rate(model, X_test, Y_test)
