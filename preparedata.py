import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
from PIL import ImageOps
from pathlib import Path

K.set_image_dim_ordering('th')

# fix random seed for reproductibility
seed = 7
np.random.seed(seed)

def get_and_prepare_data_mnist():
    
    # load data from MNIST

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # reshape to [depth][input_depth][rows][cols]

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize input from 0-255 to 0-1

    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs

    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    return (X_train, Y_train), (X_test, Y_test), num_classes

# This fonction prepare a picture for being processed by CNN
def get_and_prepare_custom_data(filename):
    #Opening in grayscale and resizing picture
    img = Image.open(filename, "r")\
        .convert('L').resize((28, 28), Image.ANTIALIAS)
    #Inverting color map
    img = ImageOps.invert(img)
    #Matrix formating
    prepared = np.asarray(img).reshape(1, 1, 28, 28).astype('float32')
    prepared = prepared / 255
    return prepared

#Print and return prediction on an unprepared data with specified model
def get_handwritten_digit_class(filename, model):
    target = get_and_prepare_custom_data(filename)
    prediction = model.predict_classes(target)
    print(Path(filename).stem + "\t->\t" + str(prediction[0]))
    img = Image.open(filename, "r")
    plt.imshow(img)
    return prediction

#Print and/or return multiples predictions on unprepared datas with specified model
def get_multiple_handwritten_digit_class(filenames, model, display=True):
    rows = len(filenames)
    if display :
        fig, axs = plt.subplots(rows, 3,figsize=(15,15))
    predictions = []
    for row, filename in enumerate(filenames) :
        target = get_and_prepare_custom_data(filename)
        predictions.append(model.predict_classes(target))
        if display :
            img = Image.open(filename, "r")
            arrow = Image.open("draws\\fleche.jpg", "r")
            axs[row, 0].set_axis_off()
            axs[row, 1].set_axis_off()
            axs[row, 2].set_axis_off()
            axs[row, 0].set_title(Path(filename).stem)
            axs[row, 0].imshow(img)
            axs[row, 1].imshow(arrow)
            axs[row, 2].text(0.5,0.5,'This is a\n' + str(predictions[row][0]), horizontalalignment='center',
             verticalalignment='center', fontsize=18, color='r')
    return predictions