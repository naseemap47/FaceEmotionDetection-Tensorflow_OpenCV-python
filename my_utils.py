import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
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


