import pandas as pd 
import numpy as np
import time
import os
import glob
import tensorflow as tf

import shutil

from sklearn.utils import class_weight

from kappa import quadratic_kappa

# label_df = pd.read_csv("trainLabels.csv")

# label_df["f"] = [j.split("_")[0] for j in label_df["image"]]
# label_df["side"] = [j.split("_")[1] for j in label_df["image"]]

# g = label_df.groupby("f")["level"].count()
# y = label_df.groupby("level").count()
# print(y)






IMG_WIDTH = 100
IMG_HEIGHT = 100



################
## PHASE 1
################
# data_dir = "Train"
INITIAL_SAMPLE = 17 #6.9%
SELECT_N = 10
img_files = glob.glob("Train/Train*/*")
img_files = img_files[:100]

unique_label = 0
training_set = pd.DataFrame()
while unique_label < 3:
    # 1. Sample 17 images
    x = np.random.choice(img_files, size=INITIAL_SAMPLE, replace = False)
    loaded_imgs = [load_img(j, IMG_HEIGHT, IMG_WIDTH) for j in x]
    labels = [get_label_from_filename(f, label_df) for f in x]
    fdf = pd.DataFrame({"img_file": x, "label": labels})
    training_set = pd.concat([training_set, fdf])

    # 2. Calculate number of unique labels
    unique_label = len(fdf.label.unique())
    print("Unique:", unique_label)
    if unique_label < 3:
        print("Repeat sampling!", unique_label)
    
print(training_set.shape)
print(training_set.label.unique())

# # 2. Compute euclidean distance between first sampled image and all other "unlabelled" images
# dist_list = []
# for img_file in img_files:
#     x = load_img(img_file)
#     d = euclidean_distance(first_img, x)
#     dist = {}    
#     dist["img_file"] = img_file
#     dist["distance"] = d
#     dist_list.append(dist)


# fdf = pd.DataFrame(dist_list)
# print(fdf.head())

# # 3. sort by distance, and remove top 2% after ranking
# fdf = fdf.sort_values("distance", ascending = False)
# idx = int(fdf.shape[0]*0.02)
# fdf = fdf.iloc[idx:,]
# print(fdf.head())

# # 4. Select top N
# selected_df = fdf.head(SELECT_N)

# # 5. LABEL the N images
# selected_df["label"] = selected_df["img_file"].apply(lambda x: get_label_from_filename(x, label_df))
# print(selected_df.sort_values("label", ascending = False).head(10))
# print(selected_df.shape)

# training_set = selected_df.copy()
################
## PHASE 2
################

TRAIN_DIR = "Selected_Train_1"
## DELETE TO RETRY
try:
    shutil.rmtree(TRAIN_DIR)
except:
    print(f"{TRAIN_DIR} does not exist.")

##### LOOP
# Train a model using the labelled images
os.makedirs(TRAIN_DIR, exist_ok=True)

for label in training_set["label"].unique():
    os.makedirs(f"{TRAIN_DIR}/{label}", exist_ok=True)

for img, label in zip(training_set["img_file"], training_set["label"]):
    shutil.copy2(img, f"{TRAIN_DIR}/{label}")

training_name = TRAIN_DIR
EPOCH = 300
BATCH_SIZE = 32
lr = 0.0001
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
image_generator = image_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    shuffle = True
)

def tiny_vgg16(NUM_CLASS, IMG_HEIGHT = 100, IMG_WIDTH = 100):
    my_model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='pool1'),

        # Block 2
        # Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
        # Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        # MaxPooling2D((2, 2), strides=(2, 2), name='pool2'),

        # Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(NUM_CLASS, activation='relu')
    ])

    return my_model



model = tiny_vgg16(NUM_CLASS = unique_label)
# model = xception_custom(NUM_CLASS=unique_label,  IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)
adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=adam,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

checkpoint_path = f"checkpoint/{training_name}" + "/checkpoint-{epoch:04d}.ckpt"

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

# Reduce the learning rate by a factor of 1/0.2 when the validation accuracy does not improve after 50 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode = "max", 
                        factor=0.9, patience=3)

early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            min_delta=0, patience=500, verbose=0, mode='max', baseline=None, restore_best_weights=False)

cs = image_generator.classes
class_weights = class_weight.compute_class_weight("balanced", np.unique(cs), cs)

print("CLASS WEIGHT:", class_weights)

history = model.fit(
            image_generator,
            steps_per_epoch = image_generator.samples // BATCH_SIZE,
            epochs=EPOCH,
            callbacks=[early_callback, reduce_lr, cp_callback],
            class_weight = class_weights
        )

eval_generator = image_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    shuffle = False
)
y_pred = model.predict(eval_generator)
y_true = eval_generator.classes
print(y_pred)
print(y_true)
y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)
# y_true = np.argmax(y_true, axis = 1)
score = quadratic_kappa(y_true, y_pred, unique_label)
print(score)

