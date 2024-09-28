from config import OUTPUT_FOLDER_PATH
import gc

def evaluate_model(model):
    # Record the memory and time for training the model
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    training_time = time.perf_counter() - start_time #second
    current, peak = tracemalloc.get_traced_memory() #Byte
    tracemalloc.stop()  # Stop tracing memory allocations
    training_memory = current / 1024 ** 2 #MB


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
        prediction_memory = current /1024 #KB

        per_sample_prediction_times.append(prediction_time * 1000) #ms
        per_sample_prediction_memories.append(prediction_memory)

        gc.collect()  # Collect garbage
        gc.enable()  # Enable garbage collector

    # Training metrics
    print(f"training time: {training_time:.6f} seconds ")
    print(f"training memory consumption: {training_memory:.6f} Megabytes (MB)")

    # Calculate and print aggregated metrics per sample

    print("Aggregated Metrics Per Sample:")
    Average_prediction_time = np.mean(per_sample_prediction_times)
    print(f"Average prediction time: {Average_prediction_time:.6f} miliseconds")

    Average_prediction_memory_consumption = np.mean(per_sample_prediction_memories)
    print(f"Average prediction memory consumption: {Average_prediction_memory_consumption:.6f} KB")

    # calculate accuracy, precision, recall, and F1 score for the entire test set
    y_preds = model.predict(X_test)
    y_preds = np.argmax(y_preds, axis=1)

    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_preds, average='weighted')
    f1 = f1_score(y_test, y_preds, average='weighted')

    # Print aggregated metrics for accuracy, precision, recall, and F1 score
    print("Aggregated Metrics for Accuracy, Precision, Recall, and F1 Score:")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    model.save(OUTPUT_FOLDER_PATH + 'ML_MLP.h5')
    ML_MLP_size = os.path.getsize(OUTPUT_FOLDER_PATH + 'ML_MLP.h5') /1024
    print(f"ML_MLP Size: {ML_MLP_size} KB")
    return model


# outputs_ML_MLP = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
#                                             'training_time_s', 'training_memory_consumption_MB', 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
#                                            'size_KB'],
#                                 'Values': [accuracy, precision, recall, f1, training_time, training_memory, Average_prediction_time, Average_prediction_memory_consumption, ML_MLP_size]})
#
# outputs_ML_MLP.to_csv('ML_MLP_outputs.csv', index=False)
#
# # Convert the lists to DataFrames
# df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
# df_memories = pd.DataFrame(per_sample_prediction_memories, columns=['Memory'])
#
# # Save to CSV
# df_times.to_csv('ML_MLP_PerSamplePredictionTime_ms.csv', index=False)
# df_memories.to_csv('ML_MLP_PerSampleMemory_KB.csv', index=False)
#
#
# ## Creating TinyML_MLP
# # Convert the model to TFLite
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# # Enable optimization for TFLite
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# # Convert the pruned and quantized model to TFLite
# start_time = time.perf_counter()
# tflite_model = converter.convert()
# conversion_time = time.perf_counter() - start_time
#
# # Save the model to disk
# with open('TinyML_MLP.tflite', 'wb') as f:
#     f.write(tflite_model)
#
# print(f'conversion time: {conversion_time: .6f} seconds', )
#
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path='TinyML_MLP.tflite')
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Initialize lists to store per-sample metrics
# per_sample_prediction_timesT = []
# per_sample_memory_usagesT = []
# predictions = []
#
# gc.collect()
# # Loop over each sample in the test data
# for i in range(X_test.shape[0]):
#     gc.disable()  # Disable garbage collector
#     tracemalloc.start()
#     start_time = time.perf_counter()
#
#     # Reshape the sample to have a batch dimension
#     input_data = np.expand_dims(X_test[i].astype(np.float32), axis=0)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     prediction = interpreter.get_tensor(output_details[0]['index'])[0]
#
#     prediction_time = time.perf_counter() - start_time
#     # Get memory usage from tracemalloc
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()  # Stop tracing memory allocations
#     prediction_memory = current #Byte
#
#     predictions.append(np.argmax(prediction))
#     per_sample_prediction_timesT.append(prediction_time * 1000000) #us
#     per_sample_memory_usagesT.append(prediction_memory)
#
#     gc.collect()  # Collect garbage
#     gc.enable()  # Enable garbage collector
#
# # Calculate and print aggregated metrics per sample
# print("Aggregated Metrics Per Sample:")
#
# Average_prediction_time = np.mean(per_sample_prediction_timesT)
# print(f"Average prediction time: {Average_prediction_time:.6f} microseconds")
#
# Average_prediction_memory_consumption = np.mean(per_sample_memory_usagesT)
# print(f"Average prediction memory consumption: {Average_prediction_memory_consumption:.6f} Bytes")
#
# # Print aggregated metrics for accuracy, precision, recall, and F1 score
# accuracyT = accuracy_score(y_test, predictions)
# precisionT = precision_score(y_test, predictions, average='weighted', zero_division=0)
# recallT = recall_score(y_test, predictions, average='weighted')
# f1T = f1_score(y_test, predictions, average='weighted')
#
# print("Aggregated Metrics for Accuracy, Precision, Recall, and F1 Score:")
# print(f"Accuracy: {accuracyT:.6f}")
# print(f"Precision: {precisionT:.6f}")
# print(f"Recall: {recallT:.6f}")
# print(f"F1 Score: {f1T:.6f}")
#
# # Saving the results into a csv
#
# TinyML_MLP_size = os.path.getsize('TinyML_MLP.tflite') / 1024
# print(f"TinyML_MLP Size: {TinyML_MLP_size} KB")
#
# outputs_TinyML_MLP = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
#                                             'conversion_time_s', 'Average_prediction_time_us', 'Average_prediction_memory_consumption_Bytes',
#                                                'size_KB'],
#                                 'Values': [ accuracyT, precisionT, recallT, f1T, conversion_time, Average_prediction_time, Average_prediction_memory_consumption, TinyML_MLP_size]})
#
# outputs_TinyML_MLP.to_csv('TinyML_MLP_outputs.csv', index=False)
#
# # Convert the lists to DataFrames
# df_timesT = pd.DataFrame(per_sample_prediction_timesT, columns=['Time'])
# df_memoriesT = pd.DataFrame(per_sample_memory_usagesT, columns=['Memory'])
#
# df_timesT.to_csv('TinyML_MLP_PerSamplePredictionTime_us.csv', index=False)
# df_memoriesT.to_csv('TinyML_MLP_PerSampleMemory_Bytes.csv', index=False)
#
#
# ## Creating ML_RF
# # Train the Random Forest model
#
# rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)  # default parameters
#
# gc.collect()
# tracemalloc.start()
# start_train_time = time.perf_counter()
# rf.fit(X_train, y_train)
# training_time = (time.perf_counter() - start_train_time) * 1000
# current, peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()  # Stop tracing memory allocations
#
# memory_usage_tracemalloc = current / 1024
#
#
# # Printing training time and memory
# print(f"Training time: {training_time:.6f} milliseconds")
# print(f"Training memory: {training_memory:.6f} KB")
#
# # Initialize lists to store per-sample metrics
# per_sample_prediction_times = []
# per_sample_memory_consumptions = []
# predictions = []
#
# gc.collect()
# # Loop over each sample in the test data
# for i in range(X_test.shape[0]):
#     # Convert X_test to a NumPy array if it's a DataFrame
#     input_data = X_test.iloc[i].values if isinstance(X_test, pd.DataFrame) else X_test[i]
#
#     gc.disable()  # Disable garbage collector
#     tracemalloc.start()
#     start_time = time.perf_counter()
#
#     prediction = rf.predict([input_data])
#
#     prediction_time = time.perf_counter() - start_time
#     # Get memory usage from tracemalloc
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()  # Stop tracing memory allocations
#     prediction_memory = current / 1024 #KB
#
#     predictions.append(prediction)
#     per_sample_prediction_times.append(prediction_time * 1000) #ms
#     per_sample_memory_consumptions.append(prediction_memory)
#
#     gc.collect()  # Collect garbage
#     gc.enable()  # Enable garbage collector
#
# y_pred = rf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
#
# # Print the metrics
# print("Accuracy of Random Forest:", accuracy)
# print("Precision of Random Forest:", precision)
# print("Recall of Random Forest:", recall)
# print("Average F1 of Random Forest:", f1)
#
# # Calculate and print aggregated metrics per sample
# average_prediction_time = np.mean(per_sample_prediction_times)
# average_prediction_memory_consumption = np.mean(per_sample_memory_consumptions)
# print("Aggregated Metrics Per Sample:")
# print(f"Average prediction time: {average_prediction_time:.6f} milliseconds")
# print(f"Average prediction memory consumption: {average_prediction_memory_consumption:.6f} KB")
#
# # Saving the results into a csv
#
# # Convert the lists to DataFrames
# df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
# df_memories = pd.DataFrame(per_sample_memory_consumptions, columns=['Memory'])
#
# df_times.to_csv('ML_RF_PerSamplePredictionTime_ms.csv', index=False)
# df_memories.to_csv('ML_RF_PerSampleMemory_KB.csv', index=False)
#
# joblib.dump(rf,'ML_RF.h5')
#
# ML_RF_size = os.path.getsize('ML_RF.h5') / 1024
# print(f"ML_RF Size: {ML_RF_size} KB")
#
#
# outputs_ML_RF = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
#                                             'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
#                                                'size_KB'],
#                                 'Values': [ accuracy, precision, recall, f1, training_time, training_memory, average_prediction_time, average_prediction_memory_consumption, ML_RF_size]})
#
# outputs_ML_RF.to_csv('ML_RF_outputs.csv', index=False)
#
#
# ## Creatig TinyML_RF
# # Get feature importances and sort them in descending order
# feature_importances = rf.feature_importances_
# indices = np.argsort(feature_importances)[::-1]
#
# feature_names = X.columns
#
# # Use a horizontal bar chart to plot the importance scores of all features in descending order. Add appropriate x-axis and y-axis labels.
# f_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
#
# # Select only the important features until accumulated importance reaches a threshold (e.g., 60%)
# importance_sum = 0.0
# selected_indices = []
# for i in range(X.shape[1]):
#     importance_sum += feature_importances[indices[i]]
#     selected_indices.append(indices[i])
#     if importance_sum >= 0.6:
#         break
#
# # Generate new training and test sets with the new feature set
# X_train_selected = X_train[:, selected_indices]
# X_test_selected = X_test[:, selected_indices]
#
# # Train the Tiny Random Forest model
# rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)  # reduced parameters
#
# gc.collect()
# tracemalloc.start()
# start_train_time = time.perf_counter()
# rf.fit(X_train_selected, y_train)
# training_time = (time.perf_counter() - start_train_time) * 1000
# current, peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()
# training_memory = current / 1024
#
#
# # Printing training time and memory
# print(f"Training time: {training_time:.6f} milliseconds")
# print(f"Training memory: {training_memory:.6f} KB")
#
# # Initialize lists to store per-sample metrics
# per_sample_prediction_times = []
# per_sample_memory_consumptions = []
# predictions = []
#
# gc.collect()
# # Loop over each sample in the test data
# for i in range(X_test_selected.shape[0]):
#     # Convert X_test_selected to a NumPy array if it's a DataFrame
#     input_data = X_test_selected.iloc[i].values if isinstance(X_test_selected, pd.DataFrame) else X_test_selected[i]
#
#     gc.disable()  # Disable garbage collector
#     tracemalloc.start()
#     start_time = time.perf_counter()
#
#     prediction = rf.predict([input_data])
#
#     prediction_time = time.perf_counter() - start_time
#     # Get memory usage from tracemalloc
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()  # Stop tracing memory allocations
#     prediction_memory = current / 1024
#
#     predictions.append(prediction)
#     per_sample_prediction_times.append(prediction_time * 1000)
#     per_sample_memory_consumptions.append(prediction_memory)
#
#     gc.collect()  # Collect garbage
#     gc.enable()  # Enable garbage collector
#
# y_pred = rf.predict(X_test_selected)
#
# # Calculate the metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
#
# # Print the metrics
# print(f"Accuracy of Random Forest: {accuracy:.6f}")
# print(f"Precision of Random Forest: {precision:.6f}")
# print(f"Recall of Random Forest: {recall:.6f}")
# print(f"Average F1 of Random Forest: {f1:.6f}")
#
# # Calculate and print aggregated metrics per sample
# average_prediction_time = np.mean(per_sample_prediction_times)
# average_prediction_memory_consumption = np.mean(per_sample_memory_consumptions)
# print("Aggregated Metrics Per Sample:")
# print(f"Average prediction time: {average_prediction_time:.6f} milliseconds")
# print(f"Average prediction memory consumption: {average_prediction_memory_consumption:.6f} KB")
#
#
# # Saving the results into a csv
#
# # Convert the lists to DataFrames
# df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
# df_memories = pd.DataFrame(per_sample_memory_consumptions, columns=['Memory'])
#
# df_times.to_csv('TinyML_RF_PerSamplePredictionTime_ms.csv', index=False)
# df_memories.to_csv('TinyML_RF_PerSampleMemory_KB.csv', index=False)
#
# joblib.dump(rf,'TinyML_RF.h5')
#
# TinyML_RF_size = os.path.getsize('TinyML_RF.h5') / 1024
# print(f"TinyML_RF Size: {TinyML_RF_size} KB")
#
#
# outputs_TinyML_RF = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
#                                             'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
#                                                'size_KB'],
#                                 'Values': [ accuracy, precision, recall, f1, training_time, training_memory, average_prediction_time, average_prediction_memory_consumption, TinyML_RF_size]})
#
# outputs_TinyML_RF.to_csv('TinyML_RF_outputs.csv', index=False)