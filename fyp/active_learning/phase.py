import os
import glob
import numpy as np
import pandas as pd
from preprocessing.load import load_img, load_label, get_label_from_filename

from cnn.xception import xception_custom
from cnn.models import small_vgg16

from active_learning.phase_1 import phase_1
from active_learning.phase_1 import phase_1
from active_learning.phase_2 import create_phase_2_dir, copy_images_to_train_dir, create_image_generator_for_training, create_image_generator_for_evaluation
from active_learning.phase_2 import initialize_model, initialize_callbacks, get_class_weight
from active_learning.cluster import compute_centroid_dict, predict_and_compute_distance_in_batch, construct_new_training_set
from active_learning.load_features_layer import load_features_model, final_4d_layer
from custom_math.kappa import quadratic_kappa

def do_phase_1(TRAIN_DIR, INITIAL_SAMPLE, IMG_HEIGHT, IMG_WIDTH):
     # import configuration

    all_filenames = glob.glob(TRAIN_DIR)

    label_df = load_label()
    training_set, unique_label = phase_1(INITIAL_SAMPLE=INITIAL_SAMPLE, TRAIN_DIR=TRAIN_DIR, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, label_df=label_df, LIMIT_DEBUG = None)

    return training_set, label_df, all_filenames


def do_phase_2(PHASE_2, ITERATION, training_set, label_df, all_filenames, EVERY_SAMPLE,
                NUM_CLASS, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, lr, EPOCH, LIMIT = None):
    # import shutil
    # try:
    #     shutil.rmtree(PHASE_2)
    # except:
    #     print("no: ", PHASE_2)

    # create phase 2 directories
    output_dir, train_dir, checkpoint_dir, output_dist_dir = create_phase_2_dir(name = PHASE_2, iteration = ITERATION)
    training_set.to_csv(f"{output_dir}/training_set.csv")
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
    # full_eval_image_generator = create_image_generator_for_evaluation("../all_train", IMG_WIDTH, IMG_HEIGHT, 128)

    ## Predict on features layer - get features for each labelled training sample
    fmodel = load_features_model(model)
    features = fmodel.predict(eval_image_generator) # compute features from last 4d layers in xception cnn model
    labels = eval_image_generator.classes

    # Compute the centroid of the features of the LABELLED samples
    updated_centroid_dict = compute_centroid_dict(features, labels)

    ## Predict on features layer - get features for each unlabelled samples
    full_dist_df = predict_and_compute_distance_in_batch(fmodel, all_filenames=all_filenames, 
                                                            centroid_dict=updated_centroid_dict, 
                                                            output_folder=output_dist_dir,
                                                            LIMIT = LIMIT)
    full_dist_df["base_img"] = full_dist_df["img_file"].apply(lambda x: os.path.basename(x))

    new_training_set = construct_new_training_set(training_set, full_dist_df, label_df = label_df, remove_top = 0.02, EVERY_SAMPLE = EVERY_SAMPLE)
    new_training_set.to_csv(f'{output_dir}/new_training_set.csv')

    return new_training_set

