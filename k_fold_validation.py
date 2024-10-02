import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import OUTPUT_FOLDER_PATH
from utils.create_df import get_final_df, clean_df
from utils.create_model import create_mlp_model



# Did not get good results with simple multilayer perceptron model

def k_fold_cross_validation():
    # Read the dataset
    df = get_final_df()

    # Clean the dataset
    df_ = clean_df(df)

    # Ensure that we have a label column
    X = df_.drop('Label', axis=1)  # Exclude the label column
    y = df_['Label']  # Include the label column

    # KFold initialization
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store results for each fold
    results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X,y)):
        print(f"Processing Fold {fold + 1}...")

        # Split the data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create and compile the model
        model = create_mlp_model(X_train, y_train)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Define early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )

        # Train the model
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0,
                  validation_split=0.1, callbacks=[early_stopping])

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=1)

        # Store results
        results.append({
            'Fold': fold + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

        print(
            f"Fold {fold + 1} - Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1:.6f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FOLDER_PATH+'k_fold_results_mlp.csv', index=False)

    # Print mean metrics
    print("\nMean Metrics:")
    print(results_df.mean(numeric_only=True))


# Call the function to execute K-Fold Cross-Validation
k_fold_cross_validation()

#
# def k_fold_cross_validation():
#     # Read the dataset
#     df = get_final_df()
#
#     # Clean the dataset
#     df_ = clean_df(df)
#
#     # Ensure that we have a label column
#     X = df_.drop('Label', axis=1)  # Exclude the label column
#     y = df_['Label']  # Include the label column
#
#     # KFold initialization
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     # Store results for each fold
#     results = []
#
#     for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
#         print(f"Processing Fold {fold + 1}...")
#
#         # Split the data into train and test sets
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#         # Create and train the Random Forest model
#         model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
#         model.fit(X_train, y_train)
#
#         # Make predictions
#         y_pred = model.predict(X_test)
#
#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
#         recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
#         f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
#
#         # Store results
#         results.append({
#             'Fold': fold + 1,
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'F1 Score': f1
#         })
#
#         print(
#             f"Fold {fold + 1} - Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1:.6f}")
#
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(OUTPUT_FOLDER_PATH + 'k_fold_results.csv', index=False)
#
#     # Print mean metrics
#     print("\nMean Metrics:")
#     print(results_df.mean(numeric_only=True))
#
#
# # Call the function to execute K-Fold Cross-Validation
# k_fold_cross_validation()

