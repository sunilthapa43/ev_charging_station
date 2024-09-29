import tensorflow as tf
import time
from config import TINYML_OUTPUT_FOLDER_PATH


def convert_to_tinyml(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # convert the model
    start_time = time.perf_counter()
    tflite_model = converter.convert()
    conversion_time = time.perf_counter() - start_time

    # save the model to disk
    with open(TINYML_OUTPUT_FOLDER_PATH+'model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"conversion_time: {conversion_time: .6f} seconds", )
    # write into the file so that we can yse it later
    with open(TINYML_OUTPUT_FOLDER_PATH+'conversion_time.txt', 'w') as f:
        f.write(f"conversion_time: {conversion_time:.6f} seconds")

    interpreter = tf.lite.Interpreter(model_path=TINYML_OUTPUT_FOLDER_PATH+"model.tflite")
    interpreter.allocate_tensors()

    return interpreter, conversion_time