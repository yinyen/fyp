import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Subtract, Concatenate, Dot
from tensorflow.keras.applications.xception import Xception


def small_vgg16(NUM_CLASS, IMG_HEIGHT = 100, IMG_WIDTH = 100):
    my_model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='pool1'),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='pool2'),

        # Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(NUM_CLASS, activation='relu')
    ])

    return my_model