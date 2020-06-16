import pandas as pd 
import numpy as np
import time
import os
import glob
import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Subtract, Concatenate, Dot


df = pd.read_csv("trainLabels.csv")

df["f"] = [j.split("_")[0] for j in df["image"]]
df["side"] = [j.split("_")[1] for j in df["image"]]

g = df.groupby("f")["level"].count()
print(g.describe())

def load_img(img_file, IMG_HEIGHT = 200, IMG_WIDTH = 200):
    img = tf.keras.preprocessing.image.load_img(img_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img / 255.0

def euclidean_distance(x, y):
    dist = np.linalg.norm(x-y)
    return dist


BATCH_SIZE = 128

# data_dir = "Train"
img_files = glob.glob("Train/Train*/*")
img_files = img_files[:100]

# 1. Sample 1 image
x = np.random.choice(img_files)
first_img = load_img(x)
print("FIRSTSHAPE:", first_img.shape)
# 2. Compute euclidean distance between first sampled image and all other "unlabelled" images

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
image_generator = image_datagen.flow_from_directory(
    'Train',
    target_size = (200, 200),
    batch_size = BATCH_SIZE,
    shuffle = False
)

x2,_ = next(image_generator)
print(x2.shape)

def data_generator(first_img):
    while True:
        input_1, _ = next(image_generator) # 32, 200, 200, 3
        input_2 = np.array([first_img]*BATCH_SIZE)  # 200, 200, 3
        # yield {"input_0": input_1, "input_1": input_2}
        yield input_1, input_2
        
# def euc(IMG_HEIGHT = 200, IMG_WIDTH = 200, CHANNEL = 3):
#     x1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNEL), name = "input_0")
#     x2 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNEL), name = "input_1")

#     x3 = Subtract()([x1, x2])
#     x3 = Flatten()(x3)
#     # x4 = Dot(axes = 0)([x3, x3])

#     model = Model(inputs=[x1, x2], outputs=x3)

#     return model

# 32, 200, 200, 3
# 32, 200*200*3

def run_dist(limit = 10):
    g = data_generator(first_img)
    z = []
    for i in range(limit):
        X1, X2 = next(g)
        print(X1.shape)
        x1 = X1.reshape((BATCH_SIZE, 200*200*3))
        x2 = X2.reshape((BATCH_SIZE, 200*200*3))
        y = x1 - x2
        y2 = y *y   
        y3 = np.sum(y2, axis = 1)
        a = y3.tolist()
        z.extend(a)
    return z

distsq = run_dist(limit = 1)

pd.DataFrame()



# g = data_generator(first_img)
# model = euc()
# q = model.predict(g)
# print(q)

# t0 = time.time()
# dist = []
# for img_file in img_files:
#     loaded_img = load_img(img_file)
#     d = euclidean_distance(first_img, loaded_img)
#     dist.append(d)

# df = pd.DataFrame({"img_file": img_files, "distance": dist})
# print(df.sort_values("distance", ascending=False))

# t1 = time.time()
# print("time:", t1-t0, (t1-t0)*35000/100)

