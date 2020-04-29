import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Subtract, Concatenate

def final_4d_layer(fmodel):
    for layer in fmodel.layers[::-1]:
        if len(layer.output_shape) == 4:
            return layer
        
def load_features_model(model):
    intermediate_layer_model = Model(inputs = model.input, outputs = final_4d_layer(model).output)
    
    return intermediate_layer_model