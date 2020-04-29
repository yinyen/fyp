import glob
import numpy as np
import pandas as pd

from preprocessing.load import load_img, load_label, get_label_from_filename
from cnn.xception import xception_custom
from cnn.models import small_vgg16
from active_learning.phase_1 import phase_1
from active_learning.phase_2 import create_phase_2_dir, copy_images_to_train_dir, create_image_generator_for_training, create_image_generator_for_evaluation
from active_learning.phase_2 import initialize_model, initialize_callbacks, get_class_weight
from active_learning.cluster import compute_centroid_dict
from custom_math.kappa import quadratic_kappa
from active_learning.load_features_layer import load_features_model, final_4d_layer
################
## PHASE 1
################

IMG_WIDTH = 200
IMG_HEIGHT = 200
NUM_CLASS = 5

INITIAL_SAMPLE = 17 #6.9%
TRAIN_DIR = "../all_train/*/*.jpeg"

label_df = load_label()
training_set, unique_label = phase_1(INITIAL_SAMPLE=INITIAL_SAMPLE, TRAIN_DIR=TRAIN_DIR, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, label_df=label_df, LIMIT_DEBUG = None)
print(training_set.shape)

SELECT_N = 10

################
## PHASE 2
################
PHASE_2 = "Phase_2_Output_v1_test5"

import shutil
try:
    shutil.rmtree(PHASE_2)
except:
    print("no: ", PHASE_2)

ITERATION = 1

BATCH_SIZE = 32
EPOCH = 6
lr = 0.0001

# create phase 2 directories
train_dir, checkpoint_dir = create_phase_2_dir(name = PHASE_2, iteration = ITERATION)

# copy labelled images to phase 2 training directory
copy_images_to_train_dir(train_dir, training_set)

# create image generator to flow images to keras model fitting procedure
image_generator = create_image_generator_for_training(train_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

##### LOOP
# Initialize model
model = xception_custom(NUM_CLASS=NUM_CLASS, IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH)
model = initialize_model(model, lr = lr)
callback_list = initialize_callbacks(checkpoint_dir)
class_weights = get_class_weight(image_generator)
print("CLASS WEIGHT:", class_weights)

# Train model
history = model.fit(
            image_generator,
            steps_per_epoch = image_generator.samples // BATCH_SIZE,
            epochs=EPOCH,
            callbacks=callback_list,
            class_weight = class_weights
        )

# Evaluate model
eval_image_generator = create_image_generator_for_evaluation(train_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)
full_eval_image_generator = create_image_generator_for_evaluation("../all_train", IMG_WIDTH, IMG_HEIGHT, 128)

## Predict on features layer - get features for each labelled training sample
print(training_set.head())
fmodel = load_features_model(model)
features = fmodel.predict(eval_image_generator, workers=4) # compute features from last 4d layers in xception cnn model
# filenames = eval_image_generator.filenames
labels = eval_image_generator.classes

# Compute the centroid of the features of the LABELLED samples
updated_centroid_dict = compute_centroid_dict(features, labels)

## Predict on features layer - get features for each unlabelled samples
import joblib

model.save_weights(checkpoint_dir + "/checkpoint.ckpt")
# joblib.dump(model, "model.pkl")
# joblib.dump(full_eval_image_generator, "full_eval_image_generator.pkl")


# joblib.dump(features, filename = "features.pkl")
# joblib.dump(filenames, filename = "filenames.pkl")
# joblib.dump(labels, filename = "labels.pkl")




# y_pred = model.predict(eval_image_generator)

# Z = model.evaluate(full_eval_image_generator, workers=4, use_multiprocessing=True)
# print(Z)

# print("===============================")
# print(y_pred)
# y_true = eval_image_generator.classes
# y_pred = np.argmax(y_pred, axis = 1)
# print(y_true, y_pred)
# score = quadratic_kappa(y_true, y_pred, 5)
# print(score)

# fmodel = load_features_model(model)

# F = fmodel.predict(eval_image_generator)
# # print(F)
# print(F.shape)