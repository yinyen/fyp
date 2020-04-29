import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Subtract, Concatenate

def final_4d_layer(fmodel):
    last_layer = None
    for layer in fmodel.layers[::-1]:
        if len(layer.output_shape) == 4:
            last_layer = layer
            app_model = 0
    if last_layer is None:
        for layer in fmodel.layers[0].layers[::-1]:
            if len(layer.output_shape) == 4:
                last_layer = layer
                app_model = 1
    return last_layer, app_model

def load_features_model(model):
    last_layer, app_model = final_4d_layer(model)
    if app_model == 1:
        intermediate_layer_model = Model(inputs = model.layers[0].input, outputs = last_layer.output)
    else:
        intermediate_layer_model = Model(inputs = model.input, outputs = last_layer.output)
    
    return intermediate_layer_model