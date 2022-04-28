from my_utils import sample_images
from my_utils import create_generators
from deeplearning_model import deeplearning_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import os
from my_utils import resize_images

################
# Switches
Sample = False
RESIZE = True
TRAIN = False
SAVE = False
################


if Sample:
    train_img_path = '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/images/train/angry/'
    sample_images(train_img_path)

if RESIZE:
    resize_images(
        to_size=256,
        path_data='/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/images'
    )

train_generators, test_generators = create_generators(
    batch_size=32,
    path_to_train_data='/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/images/train',
    path_to_val_data='/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/images/validation'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_delta=0.0001,
    verbose=1
)

if TRAIN:
    model = deeplearning_model(7)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generators,
        batch_size=32,
        epochs=50,
        validation_data=test_generators,
        callbacks=[
            early_stopping,
            learning_rate
        ]
    )

if SAVE:
    # Save Model in a h5 format
    if os.path.isfile(
            '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python//Model.h5'
    ) is False:
        model.save(
            '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python//Model.h5'
        )
