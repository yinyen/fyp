#https://www.kaggle.com/kmader/inceptionv3-for-retinopathy-gpu-hr#Split-Data-into-Training-and-Validation

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
from skimage.io import imread
import os
from glob import glob

# base_image_dir = '/media/workstation/Storage/Test/fp/train'
# retina_df = pd.read_csv(os.path.join('../', 'trainLabels.csv'))
# retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
# retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,
#                                                          '{}.jpeg'.format(x)))
# retina_df['exists'] = retina_df['path'].map(os.path.exists)
# print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
# retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
# from keras.utils.np_utils import to_categorical
# retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))

# retina_df.dropna(inplace = True)
# retina_df = retina_df[retina_df['exists']]
# retina_df.sample(3)

# retina_df[['level', 'eye']].hist(figsize = (10, 5))

# from sklearn.model_selection import train_test_split
# rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
# train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
#                                    test_size = 0.25, 
#                                    random_state = 2018,
#                                    stratify = rr_df['level'])
# raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
# valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
# print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

# train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(75, replace = True)
#                                                       ).reset_index(drop = True)
# print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
# train_df[['level', 'eye']].hist(figsize = (10, 5))

# import tensorflow as tf
# from keras import backend as K
# from keras.applications.inception_v3 import preprocess_input
# import numpy as np
# IMG_SIZE = (512, 512) # slightly smaller than vgg16 normally expects


# def tf_image_loader(out_size, 
#                       horizontal_flip = True, 
#                       vertical_flip = False, 
#                      random_brightness = True,
#                      random_contrast = True,
#                     random_saturation = True,
#                     random_hue = True,
#                       color_mode = 'rgb',
#                        preproc_func = preprocess_input,
#                        on_batch = False):
#     def _func(X):
#         with tf.name_scope('image_augmentation'):
#             with tf.name_scope('input'):
#                 # X = tf.keras.preprocessing.image.load_img(X)
#                 X = tf.image.decode_png(tf.read_file(X), channels = 3 if color_mode == 'rgb' else 0)
#                 X = tf.image.resize_images(X, out_size)
#             with tf.name_scope('augmentation'):
#                 if horizontal_flip:
#                     X = tf.image.random_flip_left_right(X)
#                 if vertical_flip:
#                     X = tf.image.random_flip_up_down(X)
#                 if random_brightness:
#                     X = tf.image.random_brightness(X, max_delta = 0.1)
#                 if random_saturation:
#                     X = tf.image.random_saturation(X, lower = 0.75, upper = 1.5)
#                 if random_hue:
#                     X = tf.image.random_hue(X, max_delta = 0.15)
#                 if random_contrast:
#                     X = tf.image.random_contrast(X, lower = 0.75, upper = 1.5)
#                 return preproc_func(X)
#     if on_batch: 
#         # we are meant to use it on a batch
#         def _batch_func(X, y):
#             return tf.map_fn(_func, X), y
#         return _batch_func
#     else:
#         # we apply it to everything
#         def _all_func(X, y):
#             return _func(X), y         
#         return _all_func
    
# def tf_augmentor(out_size,
#                 intermediate_size = (640, 640),
#                  intermediate_trans = 'crop',
#                  batch_size = 16,
#                    horizontal_flip = True, 
#                   vertical_flip = False, 
#                  random_brightness = True,
#                  random_contrast = True,
#                  random_saturation = True,
#                     random_hue = True,
#                   color_mode = 'rgb',
#                    preproc_func = preprocess_input,
#                    min_crop_percent = 0.001,
#                    max_crop_percent = 0.005,
#                    crop_probability = 0.5,
#                    rotation_range = 10):
    
#     load_ops = tf_image_loader(out_size = intermediate_size, 
#                                horizontal_flip=horizontal_flip, 
#                                vertical_flip=vertical_flip, 
#                                random_brightness = random_brightness,
#                                random_contrast = random_contrast,
#                                random_saturation = random_saturation,
#                                random_hue = random_hue,
#                                color_mode = color_mode,
#                                preproc_func = preproc_func,
#                                on_batch=False)
#     def batch_ops(X, y):
#         batch_size = tf.shape(X)[0]
#         with tf.name_scope('transformation'):
#             # code borrowed from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
#             # The list of affine transformations that our image will go under.
#             # Every element is Nx8 tensor, where N is a batch size.
#             transforms = []
#             identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
#             if rotation_range > 0:
#                 angle_rad = rotation_range / 180 * np.pi
#                 # angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
#                 angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
#                 transforms += [tf.contrib.image.angles_to_projective_transforms(angles, intermediate_size[0], intermediate_size[1])]

#             if crop_probability > 0:
#                 crop_pct = tf.random_uniform([batch_size], min_crop_percent, max_crop_percent)
#                 left = tf.random_uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
#                 top = tf.random_uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
#                 crop_transform = tf.stack([
#                       crop_pct,
#                       tf.zeros([batch_size]), top,
#                       tf.zeros([batch_size]), crop_pct, left,
#                       tf.zeros([batch_size]),
#                       tf.zeros([batch_size])
#                   ], 1)
#                 coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), crop_probability)
#                 transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
#             if len(transforms)>0:
#                 X = tf.contrib.image.transform(X,
#                       tf.contrib.image.compose_transforms(*transforms),
#                       interpolation='BILINEAR') # or 'NEAREST'
#             if intermediate_trans=='scale':
#                 X = tf.image.resize_images(X, out_size)
#             elif intermediate_trans=='crop':
#                 X = tf.image.resize_image_with_crop_or_pad(X, out_size[0], out_size[1])
#             else:
#                 raise ValueError('Invalid Operation {}'.format(intermediate_trans))
#             return X, y
#     def _create_pipeline(in_ds):
#         batch_ds = in_ds.map(load_ops, num_parallel_calls=4).batch(batch_size)
#         return batch_ds.map(batch_ops)
#     return _create_pipeline
    
# def flow_from_dataframe(idg, 
#                         in_df, 
#                         path_col,
#                         y_col, 
#                         shuffle = True, 
#                         color_mode = 'rgb'):
#     files_ds = tf.data.Dataset.from_tensor_slices((in_df[path_col].values, 
#                                                    np.stack(in_df[y_col].values,0)))
#     in_len = in_df[path_col].values.shape[0]
#     while True:
#         if shuffle:
#             files_ds = files_ds.shuffle(in_len) # shuffle the whole dataset
#         next_batch = idg(files_ds).repeat().make_one_shot_iterator().get_next()
#         for i in range(max(in_len//32,1)):
#             # NOTE: if we loop here it is 'thread-safe-ish' if we loop on the outside it is completely unsafe
#             yield K.get_session().run(next_batch)

# batch_size = 48
# core_idg = tf_augmentor(out_size = IMG_SIZE, 
#                         color_mode = 'rgb', 
#                         vertical_flip = True,
#                         crop_probability=0.0, # crop doesn't work yet
#                         batch_size = batch_size) 
# valid_idg = tf_augmentor(out_size = IMG_SIZE, color_mode = 'rgb', 
#                          crop_probability=0.0, 
#                          horizontal_flip = False, 
#                          vertical_flip = False, 
#                          random_brightness = False,
#                          random_contrast = False,
#                          random_saturation = False,
#                          random_hue = False,
#                          rotation_range = 0,
#                         batch_size = batch_size)

# train_gen = flow_from_dataframe(core_idg, train_df, 
#                              path_col = 'path',
#                             y_col = 'level_cat')

# valid_gen = flow_from_dataframe(valid_idg, valid_df, 
#                              path_col = 'path',
#                             y_col = 'level_cat') # we can use much larger batches for evaluation

# print(train_df)
# print(valid_df)
# t_x, t_y = next(train_gen)
# fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
# for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
#     c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
#     c_ax.set_title('Severity {}'.format(np.argmax(c_y, -1)))
#     c_ax.axis('off')

# print("+++++++++++++++++++++++++")
# print(t_x.shape, t_y.shape)
# print("+++++++++++++++++++++++++")
# print("+++++++++++++++++++++++++")
import numpy as np
import tensorflow as tf
from kg.init import init_tf

init_tf(device = 0)

batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
        '../all_train/train',
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode='categorical')
val_gen = val_datagen.flow_from_directory(
        '../all_train/val',
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode='categorical')

from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

def top_2_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

from kg.model import get_model

retina_model = get_model()
retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy', top_2_accuracy])
retina_model.summary()

weight_path="{}_weights.best.hdf5".format('retina')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


retina_model.fit_generator(train_gen, 
                           steps_per_epoch = train_gen.samples//batch_size,
                           validation_data = val_gen, 
                           validation_steps = val_gen.samples//batch_size,
                              epochs = 25, 
                              callbacks = callbacks_list
                            #  workers = 0, # tf-generators are not thread-safe
                            #  use_multiprocessing=False
                            #  max_queue_size = 0
                            )

# retina_model.load_weights(weight_path)
# retina_model.save('full_retina_model.h5')

# valid_gen = flow_from_dataframe(valid_idg, valid_df, 
#                              path_col = 'path',
#                             y_col = 'level_cat') 
# vbatch_count = (valid_df.shape[0]//batch_size-1)
# out_size = vbatch_count*batch_size
# test_X = np.zeros((out_size,)+t_x.shape[1:], dtype = np.float32)
# test_Y = np.zeros((out_size,)+t_y.shape[1:], dtype = np.float32)
# for i, (c_x, c_y) in zip(range(vbatch_count), valid_gen):
#     j = i*batch_size
#     test_X[j:(j+c_x.shape[0])] = c_x
#     test_Y[j:(j+c_x.shape[0])] = c_y

# for attn_layer in retina_model.layers:
#     c_shape = attn_layer.get_output_shape_at(0)
#     if len(c_shape)==4:
#         if c_shape[-1]==1:
#             print(attn_layer)
#             break

# from sklearn.metrics import accuracy_score, classification_report
# pred_Y = retina_model.predict(test_X, batch_size = 32, verbose = True)
# pred_Y_cat = np.argmax(pred_Y, -1)
# test_Y_cat = np.argmax(test_Y, -1)
# print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
# print(classification_report(test_Y_cat, pred_Y_cat))