from keras import layers
from keras import Model
import tensorflow as tf

# Pre-Trained Model (mobilenetv2_1.00_224)
model = tf.keras.applications.MobileNetV2()
# model.summary()


def deeplearning_model(no_class):
    # Input Size = 224 x 224 (pre-Trained model)
    my_input = model.layers[0].input

    # pre-trained model
    # Removing last layer in pre-trained model (it's for 1000 classes)
    # Changes to 7 classes (Our Need)
    output = model.layers[-2].output

    x = layers.Dense(128)(output)
    x = layers.Activation('relu')(x)
    x = layers.Dense(64)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(no_class, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

# new_model = deeplearning_model(7)
# new_model.summary()
