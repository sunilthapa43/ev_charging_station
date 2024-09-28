import os
import gc
import time
import datetime

from evaluation import evaluate_model
from utils.create_model import create_mlp_model
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import tracemalloc

from utils.create_df import get_final_df

df = get_final_df()
# Count the occurrences of each label
value_counts = df['Label'].value_counts()

# Calculate the total number of samples
total_number_of_samples = len(df)

# Calculate the percentage
percentages = (value_counts / total_number_of_samples) * 100

# Create a DataFrame to display both counts and percentages
New_Distribution_Results = pd.DataFrame({'Value Counts':value_counts,'Percentages':percentages})

#Getting the categories of the Labels
print('Labels are: ', df['Label'].unique().tolist())
print('The number of categories is: ', df['Label'].nunique())

## Creating our dataset
# Working with only 5 percent of the dataset due to heavy computations
print("\nLabel distribution in the full dataset: \n", df['Label'].value_counts(normalize=False))

# Create a stratified sample
df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=0.05))
df.to_csv('SampleDataset.csv')

# Check the distribution of labels in the sample
print("\nLabel distribution in the sample: \n", df['Label'].value_counts(normalize=False))

## Preprocessings
# Encoding the features
features_to_encode = ['Flow ID', 'Source IP', 'Destination IP', 'Protocol', 'Label']
encoder_dict = {}

for feature in features_to_encode:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    encoder_dict[feature] = le

#  'Timestamp' column will be replaced with datetime objects
def try_parsing_date(text):
    for fmt in ('%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M'):
        try:
            timestamp = datetime.datetime.strptime(text, fmt)
            dayofweek = timestamp.weekday()
            hour = timestamp.hour
            return pd.Series([timestamp, dayofweek, hour])
        except ValueError:
            pass
    return pd.Series([np.nan, np.nan, np.nan])

# Convert 'Timestamp' to string before applying the function
df[['Timestamp', 'DayOfWeek', 'Hour']] = df['Timestamp'].astype(str).apply(try_parsing_date)

# Now drop the 'Timestamp' column
df = df.drop(columns=['Timestamp'])

print('The number of features after preprocessing is: ', len(df.columns)-1)

# Finding the colmuns having NaN values
nan_features = df.columns[df.isna().any()].tolist()
print("Features with NaN values:", nan_features)

# Dropping any sample containing NaN or missing value
df = df.dropna()

# Replacing the infinite values with a very large number
for column in df.columns:
    if (df[column] == np.inf).any():
        df[column].replace([np.inf], [np.finfo('float32').max], inplace=True)

# Split the data into features (X) and labels (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the scaler
scaler = StandardScaler()

# Fit on the training dataset
scaler.fit(X_train)

# Transform both the training and testing data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv('X_train.csv', header=False, index=False)

X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv('X_test.csv', header=False, index=False)

y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv('y_train.csv', header=False, index=False)

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv('y_test.csv', header=False, index=False)


## Creating ML_MLP
# Create a Sequential model
model = create_mlp_model(X_train, y_train)


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # we will monitor validation loss
    patience=10,  # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # it keeps the best weights once stopped
)

# call evalauate model -> f1 sore, memory utilization, time taken, precision, recall
model = evaluate_model(model)

# now write the outputs into folder
