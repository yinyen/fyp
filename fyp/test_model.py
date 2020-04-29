import glob
import numpy as np
import pandas as pd

from preprocessing.load import load_img, load_label, get_label_from_filename
from cnn.xception import xception_custom
from cnn.models import small_vgg16
from active_learning.phase_1 import phase_1
from active_learning.phase_2 import create_phase_2_dir, copy_images_to_train_dir, create_image_generator_for_training, create_image_generator_for_evaluation
from active_learning.phase_2 import initialize_model, initialize_callbacks, get_class_weight
from custom_math.kappa import quadratic_kappa
from active_learning.load_features_layer import final_4d_layer
PHASE_2 = "Phase_2_Output_v1_test6"
ITERATION = 1

BATCH_SIZE = 32
EPOCH = 6
lr = 0.0001
NUM_CLASS=5
IMG_HEIGHT, IMG_WIDTH = 200,200
full_eval_image_generator = create_image_generator_for_evaluation("../all_train", IMG_WIDTH, IMG_HEIGHT, 128)

full_eval_image_generator


# model = small_vgg16(NUM_CLASS=NUM_CLASS, IMG_HEIGHT = IMG_HEIGHT, IMG_WIDTH = IMG_WIDTH)
# model = initialize_model(model, lr = lr)
# callback_list = initialize_callbacks(checkpoint_dir)
# class_weights = get_class_weight(image_generator)
# print("CLASS WEIGHT:", class_weights)

# Train model
# history = model.fit(
#             image_generator,
#             steps_per_epoch = image_generator.samples // BATCH_SIZE,
#             epochs=EPOCH,
#             callbacks=callback_list,
#             class_weight = class_weights
#         )

# # Evaluate model
# eval_image_generator = create_image_generator_for_evaluation(train_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

# y_pred = model.predict(eval_image_generator)
# print(y_pred)
# y_true = eval_image_generator.classes
# y_pred = np.argmax(y_pred, axis = 1)
# print(y_true, y_pred)
# score = quadratic_kappa(y_true, y_pred, len(np.unique(y_true)))
# print(score)

print(model.layers)
print(model.layers[::-1])
print(final_4d_layer(model))