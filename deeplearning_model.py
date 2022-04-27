from keras import layers
from keras import Model


def deeplearning_model(no_class):
    my_input = layers.Input(shape=(48, 48, 1))

    # 1st CNN Layer
    x = layers.Conv2D(64, (3, 3), padding='same')(my_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    # 2nd CNN Layer
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    # 3rd CNN Layer
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    # 4th CNN Layer
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    # Fully connected 1st Layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)