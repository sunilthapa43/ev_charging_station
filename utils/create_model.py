import tensorflow as tf
import numpy as np


## Creating ML_MLP
# Create a Sequential model (Multilayer Perceptron)

def create_mlp_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        # Encountered UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
        # updating the code:

        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax'),  # The output layer size matches the number of classes
    ])
    return model


