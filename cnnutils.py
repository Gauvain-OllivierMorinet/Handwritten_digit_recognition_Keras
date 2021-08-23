from keras.models import model_from_json

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

# Evaluate a model using data and expected predictions
def print_model_error_rate(model, X_test, Y_test):
    # Evaluating model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    # Human readable results
    print("Model score : %.2f%%" % (scores[1] * 100))
    print("Model error rate : %.2f%%" % (100 - scores[1] * 100))

# This function saves a model on the drive using 2 files : .json an .h5
def save_keras_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        jsonfile.write(model_json)
    #serialize weights to HDF5
    model.save_weights(filename + ".h5")

# This function load a model from 2 files : .json an .h5
# WARNING : Don't mix different model's data
# The new model need to be compiled before any use
def load_keras_model(filename):
    #load JSON and create the model
    with open(filename + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model.model_from_json(loaded_model_json)
    # load weights into the new model
    loaded_model.load_weights(filename + ".h5")
    # print a summary of the loaded model
    loaded_model.summary()
    return loaded_model
