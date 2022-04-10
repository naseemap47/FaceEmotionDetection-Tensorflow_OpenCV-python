from keras import layers
from keras import Model


def deeplearning_model(no_class):
    my_input = layers.Input(shape=(48, 48, 1))

    x = layers.Conv2D(32, (3, 3), padding='same')(my_input)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)
