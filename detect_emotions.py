from my_utils import sample_images
from my_utils import create_generators
from deeplearning_model import deeplearning_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import os

train_img_path = '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/images/train/angry/'
# sample_images(train_img_path)

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
    min_delta=0.001,
    verbose=1
)

model = deeplearning_model(7)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generators,
    batch_size=32,
    epochs=2,
    validation_data=test_generators,
    callbacks=[early_stopping, learning_rate]
)

# Save Model in a h5 format

if os.path.isfile(
        '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python//Model.h5'
) is False:
    model.save(
        '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python//Model.h5'
    )