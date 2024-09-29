from config import OUTPUT_FOLDER_PATH, TINYML_OUTPUT_FOLDER_PATH
import tensorflow as tf
import os
import pandas as pd
from train_model import train_model
from utils.convert_model import convert_to_tinyml


# get the model from the file and load it

def get_model():
    if os.path.exists(OUTPUT_FOLDER_PATH + '/ML_MLP.h5'):
        print("Seems like model is already trained and ready to use")
        # load model
        model = tf.keras.models.load_model(OUTPUT_FOLDER_PATH + '/ML_MLP.h5')
        X_test = pd.read_csv(OUTPUT_FOLDER_PATH + '/X_test.csv')
        y_test = pd.read_csv(OUTPUT_FOLDER_PATH + '/y_test.csv')
    else:
        model, X_test, y_test = train_model()

    return model, X_test, y_test



def get_tinyml_model():
    if os.path.exists(TINYML_OUTPUT_FOLDER_PATH + '/model.tflite'):
        print("Seems like TINYML model is already trained and ready to use")
        # load model
        model = tf.lite.Interpreter(model_path=TINYML_OUTPUT_FOLDER_PATH + '/model.tflite')
        model.allocate_tensors()
        X_test = pd.read_csv(OUTPUT_FOLDER_PATH + '/X_test.csv')
        y_test = pd.read_csv(OUTPUT_FOLDER_PATH + '/y_test.csv')
    else:
        model, X_test, y_test = get_model()
        # convert model to TINYML
        model, conversion_time = convert_to_tinyml(model)

    return model, X_test, y_test