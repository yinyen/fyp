import glob
import numpy as np
import pandas as pd
import os
from preprocessing.load import load_img, load_label, get_label_from_filename, load_img_fast
from cnn.xception import xception_custom
from cnn.models import small_vgg16
from active_learning.phase_1 import phase_1
from active_learning.phase_2 import create_phase_2_dir, copy_images_to_train_dir, create_image_generator_for_training, create_image_generator_for_evaluation
from active_learning.phase_2 import initialize_model, initialize_callbacks, get_class_weight
from custom_math.kappa import quadratic_kappa
from active_learning.load_features_layer import final_4d_layer, load_features_model
from active_learning.cluster import compute_distance_df_per_batch, compute_centroid_dict, predict_and_compute_distance_in_batch
import joblib
import time
# PHASE_2 = "Phase_2_Output_v1_test6"
# ITERATION = 1

# BATCH_SIZE = 32
# EPOCH = 6
# lr = 0.0001
# NUM_CLASS=5
IMG_HEIGHT, IMG_WIDTH = 200,200

features = joblib.load("features.pkl")
filenames = joblib.load("filenames.pkl")
labels = joblib.load("labels.pkl")
centroid_dict = compute_centroid_dict(features, labels)

model = xception_custom(5,200,200)
model.load_weights("Phase_2_Output_v1_test5/01/Checkpoint/checkpoint.ckpt")
fmodel = load_features_model(model)

filenames = glob.glob("../all_train/*/*.jpeg")


t0 = time.time()
output_folder = "Phase_2_Output_v1_test5/01/Cluster"
fdf = predict_and_compute_distance_in_batch(fmodel, filenames, centroid_dict, output_folder)
print(fdf.shape)
print(fdf.head())
t1 = time.time()
print(t1-t0)
time.sleep(30)
# def b(all_filenames, fmodel, centroid_dict, batch = 1024):
    
#     predicted_features = fmodel.predict(x)
#     print(f"DONE: {index}")
#     df = compute_distance_df_per_batch(predicted_features, filenames, centroid_dict)
#     df.to_csv(f"{index}.csv")
#     return df





# filenames = [os.path.join("../train/0",j) for j in g0.filenames]
# df = b(0, fmodel, filenames, centroid_dict)
# print(df)
# t1 = time.time()
# print(t1-t0)

# t0 = time.time()
# inputss = [(j, fmodel, filenames[j*128:((j+1)*128)], centroid_dict) for j in range(2)]
# index_list = [j for j in range(2)]
# fmodel_list = [fmodel]*2
# filenames_list = [filenames[j*128:((j+1)*128)] for j in range(4)]
# centroid_dict_list = [centroid_dict for j in range(2)]

# p = Pool(8)
# res = p.map(load_img, filenames[:512])
# x2 = np.array(res)
# print(x2.shape)
# def add(x,y):
#     return x+y

# print(res)
# for i in range(2):
#     p.apply_async(b, inputss)
# res = p.starmap_async(b, inputss)

# p.close()
# p.join()
# print(res)


# ff = g0.filenames
# import time
# t0 = time.time()
# x,y = next(g0)
# # x = np.array([load_img(os.path.join("../train/0", f)) for f in ff[:128]])
# print(x.shape)
# t1 = time.time()
# print(t1-t0)



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

# print(model.layers)
# print(model.layers[::-1])
# print(final_4d_layer(model))