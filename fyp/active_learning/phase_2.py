import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

def create_phase_2_dir(name, iteration):
    PHASE_2 = name
    ITERATION = iteration
    ITERATION_STR = str(ITERATION).zfill(2)
    PHASE_2_TRAIN_DIR = "Train"
    PHASE_2_CHECKPOINT_DIR = "Checkpoint"
    PHASE_2_OUTPUT_DIST_DIR = "Cluster"

    os.makedirs(PHASE_2, exist_ok=False)
    os.makedirs(f'{PHASE_2}/{ITERATION_STR}')
    os.makedirs(f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_TRAIN_DIR}')
    os.makedirs(f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_CHECKPOINT_DIR}')
    os.makedirs(f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_OUTPUT_DIST_DIR}')
    for label in [0,1,2,3,4]:
        label = str(label)
        os.makedirs(f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_TRAIN_DIR}/{label}', exist_ok=False)

    output_dir = f'{PHASE_2}/{ITERATION_STR}' 
    train_dir = f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_TRAIN_DIR}'
    checkpoint_dir = f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_CHECKPOINT_DIR}'
    output_dist_dir = f'{PHASE_2}/{ITERATION_STR}/{PHASE_2_OUTPUT_DIST_DIR}'
    return output_dir, train_dir, checkpoint_dir, output_dist_dir

def copy_images_to_train_dir(train_dir, training_set):
    for img, label in zip(training_set["img_file"], training_set["label"]):
        shutil.copy2(img, f"{train_dir}/{label}")

def create_image_generator_for_training(train_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
    image_generator = image_datagen.flow_from_directory(
        train_dir,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    return image_generator

def create_image_generator_for_evaluation(train_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
    image_generator = image_datagen.flow_from_directory(
        train_dir,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        batch_size = BATCH_SIZE,
        shuffle = False
    )
    return image_generator

def initialize_model(model, lr):
    adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
    return model
    

def initialize_callbacks(checkpoint_dir):
    checkpoint_path = f"{checkpoint_dir}" + "/checkpoint-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights every 100 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_accuracy', mode = "max",
        filepath=checkpoint_path, 
        period=5,
        verbose=1, 
        save_weights_only=True,
        save_best_only=True
    )

    # Create a callback to visualize model performance while training using tensorboard
    # tb_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir='./tensorboard_logs/{}'.format(training_name), histogram_freq=0, update_freq='epoch')

    # Reduce the learning rate by a factor of 1/0.95 when the validation accuracy does not improve after 3 epochs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode = "max", 
                            factor=0.95, patience=3)

    early_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                min_delta=0, patience=50, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    callback_list = [cp_callback, reduce_lr, early_callback]
    
    return callback_list


def get_class_weight(image_generator):
    cs = image_generator.classes
    uniq = np.unique(cs)
    print(uniq)
    if len(uniq) < 5:
        cs = np.array(cs).tolist()*10
        for i in range(5):
            cs.append(i)
    class_weights = class_weight.compute_class_weight("balanced", np.unique(cs), cs)
    return class_weights


def delete_dir():
    ## DELETE TO RETRY
    # try:
    #     shutil.rmtree(TRAIN_DIR)
    # except:
    #     print(f"{TRAIN_DIR} does not exist.")
    return None