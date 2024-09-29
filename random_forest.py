import pandas as pd
import numpy as np
import gc
import tracemalloc
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.create_df import get_final_df, clean_df

## Creating ML_RF
# Train the Random Forest model
df = get_final_df()
df_ = clean_df(df)
X_train, X_test, y_train, y_test = train_test_split(df_)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)  # default parameters

gc.collect()
tracemalloc.start()
start_train_time = time.perf_counter()
rf.fit(X_train, y_train)
training_time = (time.perf_counter() - start_train_time) * 1000
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()  # Stop tracing memory allocations

training_memory = current / 1024 ** 2


# Printing training time and memory
print(f"Training time: {training_time:.6f} milliseconds")
print(f"Training memory: {training_memory:.6f} KB")

# Initialize lists to store per-sample metrics
per_sample_prediction_times = []
per_sample_memory_consumptions = []
predictions = []

gc.collect()
# Loop over each sample in the test data
for i in range(X_test.shape[0]):
    # Convert X_test to a NumPy array if it's a DataFrame
    input_data = X_test.iloc[i].values if isinstance(X_test, pd.DataFrame) else X_test[i]

    gc.disable()  # Disable garbage collector
    tracemalloc.start()
    start_time = time.perf_counter()

    prediction = rf.predict([input_data])

    prediction_time = time.perf_counter() - start_time
    # Get memory usage from tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    prediction_memory = current / 1024 #KB

    predictions.append(prediction)
    per_sample_prediction_times.append(prediction_time * 1000) #ms
    per_sample_memory_consumptions.append(prediction_memory)

    gc.collect()  # Collect garbage
    gc.enable()  # Enable garbage collector

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print("Accuracy of Random Forest:", accuracy)
print("Precision of Random Forest:", precision)
print("Recall of Random Forest:", recall)
print("Average F1 of Random Forest:", f1)

# Calculate and print aggregated metrics per sample
average_prediction_time = np.mean(per_sample_prediction_times)
average_prediction_memory_consumption = np.mean(per_sample_memory_consumptions)
print("Aggregated Metrics Per Sample:")
print(f"Average prediction time: {average_prediction_time:.6f} milliseconds")
print(f"Average prediction memory consumption: {average_prediction_memory_consumption:.6f} KB")

# Saving the results into a csv

# Convert the lists to DataFrames
df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
df_memories = pd.DataFrame(per_sample_memory_consumptions, columns=['Memory'])

df_times.to_csv('ML_RF_PerSamplePredictionTime_ms.csv', index=False)
df_memories.to_csv('ML_RF_PerSampleMemory_KB.csv', index=False)

joblib.dump(rf,'ML_RF.h5')

ML_RF_size = os.path.getsize('ML_RF.h5') / 1024
print(f"ML_RF Size: {ML_RF_size} KB")


outputs_ML_RF = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
                                            'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
                                               'size_KB'],
                                'Values': [ accuracy, precision, recall, f1, training_time, training_memory, average_prediction_time, average_prediction_memory_consumption, ML_RF_size]})

outputs_ML_RF.to_csv('ML_RF_outputs.csv', index=False)


## Creatig TinyML_RF
# Get feature importances and sort them in descending order
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]

feature_names = X.columns

# Use a horizontal bar chart to plot the importance scores of all features in descending order. Add appropriate x-axis and y-axis labels.
f_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()

# Select only the important features until accumulated importance reaches a threshold (e.g., 60%)
importance_sum = 0.0
selected_indices = []
for i in range(X.shape[1]):
    importance_sum += feature_importances[indices[i]]
    selected_indices.append(indices[i])
    if importance_sum >= 0.6:
        break

# Generate new training and test sets with the new feature set
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Train the Tiny Random Forest model
rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)  # reduced parameters

gc.collect()
tracemalloc.start()
start_train_time = time.perf_counter()
rf.fit(X_train_selected, y_train)
training_time = (time.perf_counter() - start_train_time) * 1000
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
training_memory = current / 1024


# Printing training time and memory
print(f"Training time: {training_time:.6f} milliseconds")
print(f"Training memory: {training_memory:.6f} KB")

# Initialize lists to store per-sample metrics
per_sample_prediction_times = []
per_sample_memory_consumptions = []
predictions = []

gc.collect()
# Loop over each sample in the test data
for i in range(X_test_selected.shape[0]):
    # Convert X_test_selected to a NumPy array if it's a DataFrame
    input_data = X_test_selected.iloc[i].values if isinstance(X_test_selected, pd.DataFrame) else X_test_selected[i]

    gc.disable()  # Disable garbage collector
    tracemalloc.start()
    start_time = time.perf_counter()

    prediction = rf.predict([input_data])

    prediction_time = time.perf_counter() - start_time
    # Get memory usage from tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    prediction_memory = current / 1024

    predictions.append(prediction)
    per_sample_prediction_times.append(prediction_time * 1000)
    per_sample_memory_consumptions.append(prediction_memory)

    gc.collect()  # Collect garbage
    gc.enable()  # Enable garbage collector

y_pred = rf.predict(X_test_selected)

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print(f"Accuracy of Random Forest: {accuracy:.6f}")
print(f"Precision of Random Forest: {precision:.6f}")
print(f"Recall of Random Forest: {recall:.6f}")
print(f"Average F1 of Random Forest: {f1:.6f}")

# Calculate and print aggregated metrics per sample
average_prediction_time = np.mean(per_sample_prediction_times)
average_prediction_memory_consumption = np.mean(per_sample_memory_consumptions)
print("Aggregated Metrics Per Sample:")
print(f"Average prediction time: {average_prediction_time:.6f} milliseconds")
print(f"Average prediction memory consumption: {average_prediction_memory_consumption:.6f} KB")


# Saving the results into a csv

# Convert the lists to DataFrames
df_times = pd.DataFrame(per_sample_prediction_times, columns=['Time'])
df_memories = pd.DataFrame(per_sample_memory_consumptions, columns=['Memory'])

df_times.to_csv('TinyML_RF_PerSamplePredictionTime_ms.csv', index=False)
df_memories.to_csv('TinyML_RF_PerSampleMemory_KB.csv', index=False)

joblib.dump(rf,'TinyML_RF.h5')

TinyML_RF_size = os.path.getsize('TinyML_RF.h5') / 1024
print(f"TinyML_RF Size: {TinyML_RF_size} KB")


outputs_TinyML_RF = pd.DataFrame({ 'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1_Score',
                                            'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms', 'Average_prediction_memory_consumption_KB',
                                               'size_KB'],
                                'Values': [ accuracy, precision, recall, f1, training_time, training_memory, average_prediction_time, average_prediction_memory_consumption, TinyML_RF_size]})

outputs_TinyML_RF.to_csv('TinyML_RF_outputs.csv', index=False)
