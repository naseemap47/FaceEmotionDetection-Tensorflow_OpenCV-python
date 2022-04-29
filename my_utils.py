import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
import os


def sample_images(img_folder_path):
    # x = np.random.randint(10, size=1)
    plt.figure(figsize=(5, 5))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(
            img_folder_path + os.listdir(img_folder_path)[i],
            target_size=(48, 48)
        )
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
    plt.show()


def create_generators(batch_size, path_to_train_data, path_to_val_data):
    train_preprocessor = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator()
    train_generators = train_preprocessor.flow_from_directory(
        path_to_train_data,
        target_size=(256, 256),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )
    val_generators = val_preprocessor.flow_from_directory(
        path_to_val_data,
        target_size=(256, 256),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    return train_generators, val_generators


def resize_images(to_size, path_data):
    files = os.listdir(path_data)
    for file in files:
        path = os.path.join(path_data, file)
        category = os.listdir(path)
        for cat in category:
            path_cat = os.path.join(path, cat)
            images = os.listdir(path_cat)
            for img in images:
                path_img = os.path.join(path_cat, img)
                img_array = cv2.imread(path_img)
                new_array = cv2.resize(img_array, (to_size, to_size))
                cv2.imwrite(os.path.join(path_cat, img), new_array)