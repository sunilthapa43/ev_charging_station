import pandas as pd
import tensorflow as tf
from utils.convert_model import convert_to_tinyml
from config import OUTPUT_FOLDER_PATH
from utils.evaluation import evaluate_model, evaluate_tiny_ml
from utils.create_model import create_mlp_model
from utils.create_df import get_final_df, clean_df, split_into_train_test


# create and train the model here:

def train_model():
    # First read the dataset from the file locations, convert into the dataframe and combine the dataframe, get_final_df() does the task
    df = get_final_df()

    # Now we have to filter the nan values, drop the rows with na values, keep only the features we need
    df_ = clean_df(df)

    # get the training and testing sets by
    X_train, X_test, y_train, y_test = split_into_train_test(df_)

    X_train_df = pd.DataFrame(X_train)
    X_train_df.to_csv(OUTPUT_FOLDER_PATH + 'X_train.csv', header=False, index=False)

    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(OUTPUT_FOLDER_PATH + 'X_test.csv', header=False, index=False)

    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv(OUTPUT_FOLDER_PATH + 'y_train.csv', header=False, index=False)

    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv(OUTPUT_FOLDER_PATH + 'y_test.csv', header=False, index=False)

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
        patience=5,  # number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # it keeps the best weights once stopped
    )

    # call evalauate model -> f1 sore, memory utilization, time taken, precision, recall
    model = evaluate_model(
        model, early_stopping, X_train, y_train, X_test, y_test
    )

    return model, X_test, y_test

