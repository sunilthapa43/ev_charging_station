from utils.create_df import get_final_df, clean_df
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from config import OUTPUT_FOLDER_PATH

df = get_final_df()
df_ = clean_df(df)


# get labels and features
X = df.drop('Label', axis=1)
y = df['Label']

# we are implementing 5-fold cross-validation
kf = KFold(5, shuffle=True, random_state=42)

# store each metrices
accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = tf.keras.model.load_model(OUTPUT_FOLDER_PATH + "/ML_MLP.h5")
    # compile the model with adam optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

    # predictions
    y_pred = model.predict(x_test)

    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Fold {kf.n_splits} Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1 Score: {f1:.6f}")

# calculate mean of metrics
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1_score = np.mean(f1_scores)

print(f"\nMean Metrics:")
print(f"Accuracy: {mean_accuracy:.6f}, Precision: {mean_precision:.6f}, Recall: {mean_recall:.6f}, F1 Score: {mean_f1_score:.6f}")

# print classification report
print("\nClassification report for the last fold:")
print(classification_report(y_test, y_pred_classes))