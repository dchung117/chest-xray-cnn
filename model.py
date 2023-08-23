from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers

def get_model(input_shape: tuple[int, int, int], 
        weights: Optional[str] = "imagenet",
        include_top: bool = False) -> tf.keras.Model:
    """
    Initialize InceptionV3 model for chest x-ray classification
    
    Args
    ----
        input_shape: tuple[int, int, int]
            3-element of tuple of (img_height, img_weight, channel)
        weights: Optional[str] = "imagenet"
            Optional pretrained weights to initialize InceptionV3 backbone
        include_top: bool = False
            Flag to include original head of InceptionV3
    
    Returns
    -------
        tf.keras.Model
            Initialized InceptionV3 model
    """
    inception = tf.keras.applications.inception_v3.InceptionV3(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top
    )

    # freeze pretrained layers
    for layer in inception.layers:
        layer.trainable = False

    last_layer = inception.layers[-1]
    x = layers.Flatten()(last_layer.output)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inception.input, x)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    return model