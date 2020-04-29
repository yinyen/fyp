import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Subtract, Concatenate, Dot
from tensorflow.keras.applications.xception import Xception


def xception_custom(NUM_CLASS, IMG_HEIGHT, IMG_WIDTH):
    print("XCEPTION")
    cnn = Xception(include_top = False, weights = None, 
                      input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), pooling = None)

    model = Sequential()
    model.add(cnn)
    model.add(Dense(units = 256, activation='relu'))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = NUM_CLASS, activation='relu'))
    return model


def xception_2(NUM_CLASS, IMG_HEIGHT, IMG_WIDTH):
    print("XCEPTION")
    cnn = Xception(include_top = False, weights = 'imagenet', 
                      input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), pooling = None)

    model = Sequential()
    model.add(cnn)
    model.add(Dense(units = 512, activation='relu'))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = NUM_CLASS, activation='relu'))
    return model
