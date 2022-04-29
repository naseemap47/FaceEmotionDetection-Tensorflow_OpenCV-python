from keras import layers
from keras import Model


def deeplearning_model(no_class):
    my_input = layers.Input(shape=(256, 256, 3))

    # Layer 1
    x = layers.Conv2D(16, (3, 3), activation='relu')(my_input)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    # Layer 2
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    # Layer 3
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    # Layer 4
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

# new_model = deeplearning_model(7)
# new_model.summary()
