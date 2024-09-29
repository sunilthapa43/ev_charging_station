import gc
import os
import time
import pandas as pd
import tracemalloc
import numpy as np
from config import OUTPUT_FOLDER_PATH, TINYML_OUTPUT_FOLDER_PATH
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TINY_ML_DF_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1_Score',
                      'conversion_time_s', 'Average_prediction_time_us',
                      'Average_prediction_memory_consumption_Bytes',
                      'size_KB']

ML_DF_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1_Score',
                 'training_time_s', 'training_memory_consumption_MB',
                 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
                 'size_KB']


def handle_write(y_test, per_sample_prediction_times, per_sample_memory_usages, X_test, model=None, y_preds=None,
                 training_time=None, conversion_time=None, training_memory=None):
    if model:
        y_preds = model.predict(X_test)
        y_preds = np.argmax(y_preds, axis=1)
        output_path = OUTPUT_FOLDER_PATH
        model_path = os.path.join(OUTPUT_FOLDER_PATH, "ML_MLP.h5")
        unit_time = "milliseconds"
        unit_memory_usage = "KB"
    else:
        output_path = TINYML_OUTPUT_FOLDER_PATH
        model_path = os.path.join(TINYML_OUTPUT_FOLDER_PATH, 'model.tflite')
        unit_time = "microseconds"
        unit_memory_usage = "Bytes"
    # Calculate and print aggregated metrics per sample
    print("Aggregated Metrics Per Sample:")

    Average_prediction_time = np.mean(per_sample_prediction_times)
    print(f"Average prediction time: {Average_prediction_time:.6f}" + unit_time)

    Average_prediction_memory_consumption = np.mean(per_sample_memory_usages)
    print(f"Average prediction memory consumption: {Average_prediction_memory_consumption:.6f}" + unit_memory_usage)

    # Print aggregated metrics for accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_preds, average='weighted')
    f1 = f1_score(y_test, y_preds, average='weighted')

    print("Aggregated Metrics for Accuracy, Precision, Recall, and F1 Score:")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    # Saving the results into a csv
    model.save(OUTPUT_FOLDER_PATH + 'ML_MLP.h5') if model else None
    model_size = os.path.getsize(model_path) / 1024
    print(f"Model Size: {model_size} KB")

    outputs = pd.DataFrame({'Metrics': ML_DF_METRICS if model else TINY_ML_DF_METRICS,
                            'Values': [accuracy, precision, recall, f1, training_time, training_memory,
                                       Average_prediction_time, Average_prediction_memory_consumption,
                                       model_size] if model else
                            [accuracy, precision, recall, f1, conversion_time, Average_prediction_time,
                             Average_prediction_memory_consumption, model_size]
                            })

    outputs.to_csv(output_path + "/outputs.csv", index=False)

    # Convert the lists to DataFrames
    df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
    df_memories = pd.DataFrame(per_sample_memory_usages, columns=['Memory'])

    df_times.to_csv(output_path + "/prediction_times.csv", index=False)
    df_memories.to_csv(output_path + "/per_sample_memory.csv", index=False)
    return model if model else None


def evaluate_model(model, early_stopping, X_train, y_train, X_test, y_test):
    # Record the memory and time for training the model
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    training_time = time.perf_counter() - start_time  # second
    current, peak = tracemalloc.get_traced_memory()  # Byte
    tracemalloc.stop()  # Stop tracing memory allocations
    training_memory = current / 1024 ** 2  # MB

    # Prediction
    # Initialize lists to store per-sample metrics
    per_sample_prediction_times = []
    per_sample_prediction_memories = []

    gc.collect()
    for sample_idx in range(X_test.shape[0]):
        gc.disable()  # Disable garbage collector
        tracemalloc.start()
        start_time = time.perf_counter()
        y_pred = model.predict(np.expand_dims(X_test[sample_idx], axis=0), verbose=1)
        prediction_time = time.perf_counter() - start_time
        # Get memory usage from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # Stop tracing memory allocations
        prediction_memory = current / 1024  # KB

        per_sample_prediction_times.append(prediction_time * 1000)  # ms
        per_sample_prediction_memories.append(prediction_memory)

        gc.collect()  # Collect garbage
        gc.enable()  # Enable garbage collector

    # Training metrics
    print(f"training time: {training_time:.6f} seconds ")
    print(f"training memory consumption: {training_memory:.6f} Megabytes (MB)")

    # Calculate and print aggregated metrics per sample
    model = handle_write(model=model, X_test=X_test, y_test=y_test, y_preds=y_pred, training_time=training_time,
                         per_sample_prediction_times=per_sample_prediction_times,
                         per_sample_memory_usages=per_sample_prediction_memories, training_memory=training_memory,
                         )
    return model

def evaluate_tiny_ml(interpreter, X_test, y_test):

    print(X_test, y_test)
    # # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Initialize lists to store per-sample metrics
    per_sample_prediction_timesT = []
    per_sample_memory_usagesT = []
    predictions = []

    gc.collect()

    # convert to numpy array if x_test is a dataframe
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    # Loop over each sample in the test data
    for i in range(X_test.shape[0]):
        gc.disable()  # Disable garbage collector
        tracemalloc.start()
        start_time = time.perf_counter()


        # Reshape the sample to have a batch dimension
        input_data = np.expand_dims(X_test[i].astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        prediction_time = time.perf_counter() - start_time
        # Get memory usage from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # Stop tracing memory allocations
        prediction_memory = current  # Byte

        predictions.append(np.argmax(prediction))
        per_sample_prediction_timesT.append(prediction_time * 1000000)  # us
        per_sample_memory_usagesT.append(prediction_memory)

        gc.collect()  # Collect garbage
        gc.enable()  # Enable garbage collector
    handle_write(y_test=y_test, per_sample_prediction_times=per_sample_prediction_timesT,
                     per_sample_memory_usages=per_sample_memory_usagesT,
                     X_test=X_test)
