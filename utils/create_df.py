import pandas as pd
import datetime
import numpy as np
from .constants import keep_features, DoS_Types, Brute_Force_Types, Web_Attack_types, Others
from config import DATASET_FOLDER_PATH, OUTPUT_FOLDER_PATH
from sklearn.model_selection import train_test_split
# import labelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_final_df():
    # Load the dataset
    df1 = pd.read_csv(DATASET_FOLDER_PATH + '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df2 = pd.read_csv(DATASET_FOLDER_PATH + '/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df3 = pd.read_csv(DATASET_FOLDER_PATH + '/Friday-WorkingHours-Morning.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)
    df4 = pd.read_csv(DATASET_FOLDER_PATH + '/Monday-WorkingHours.pcap_ISCX.csv', encoding='latin1', low_memory=False)
    df5 = pd.read_csv(DATASET_FOLDER_PATH + '/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                      encoding='latin1', low_memory=False)
    df6 = pd.read_csv(DATASET_FOLDER_PATH + '/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                      encoding='latin1', low_memory=False)
    df7 = pd.read_csv(DATASET_FOLDER_PATH + '/Tuesday-WorkingHours.pcap_ISCX.csv', encoding='latin1', low_memory=False)
    df8 = pd.read_csv(DATASET_FOLDER_PATH + '/Wednesday-workingHours.pcap_ISCX.csv', encoding='latin1',
                      low_memory=False)

    # Get the features
    column_names1 = df1.columns.tolist()
    print('The features are: ', column_names1)
    print('The number of features is: ', len(column_names1))

    print(
        "===========================================================================================================================================================")
    print("Output folder path is", OUTPUT_FOLDER_PATH)
    print(
        "===========================================================================================================================================================")

    ## Creating Our DataFrame
    # Merge the DataFrames horizontally based on common columns (all columns)
    combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=0)
    combined_df.head()

    # Remove spaces before the first letter of column names
    combined_df = combined_df.rename(columns=lambda x: x.strip())
    print('The number of rows in the combined dataset is: ', combined_df.shape[0])

    # The features that are related to Electric Vehicle Charging Stations are kept and the other features are removed
    # Remove the features not in 'keep_features'
    Final_df = combined_df.drop(columns=[col for col in combined_df.columns if col not in keep_features])
    final_column_names = Final_df.columns.tolist()
    print('number of selected features: ', len(final_column_names) - 1)

    # Getting the categories of the Labels
    print('Labels are: ', Final_df['Label'].unique().tolist())
    print('The number of categories is: ', Final_df['Label'].nunique())

    # Count the occurrences of each label
    value_counts = Final_df['Label'].value_counts()

    # Calculate the total number of samples
    total_number_of_samples = len(Final_df)

    # Calculate the percentage
    percentages = (value_counts / total_number_of_samples) * 100

    # Create a DataFrame to display both counts and percentages
    Distribution_Results = pd.DataFrame({'Value Counts': value_counts, 'Percentages': percentages})
    print(Distribution_Results)

    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Dos' if x in DoS_Types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Brute_Force' if x in Brute_Force_Types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Web_Attack' if x in Web_Attack_types else x)
    Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Bot/Infiltration/Heartbleed' if x in Others else x)
    return Final_df


def clean_df(df):
    # Count the occurrences of each label
    value_counts = df['Label'].value_counts()

    # Calculate the total number of samples
    total_number_of_samples = len(df)

    # Calculate the percentage
    percentages = (value_counts / total_number_of_samples) * 100

    # Create a DataFrame to display both counts and percentages
    New_Distribution_Results = pd.DataFrame({'Value Counts': value_counts, 'Percentages': percentages})

    # Getting the categories of the Labels
    print('Labels are: ', df['Label'].unique().tolist())
    print('The number of categories is: ', df['Label'].nunique())

    ## Creating our dataset
    # Working with only 5 percent of the dataset due to heavy computations
    print("\nLabel distribution in the full dataset: \n", df['Label'].value_counts(normalize=False))

    # Create a stratified sample
    df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=0.05))
    df.to_csv(OUTPUT_FOLDER_PATH + 'SampleDataset.csv')

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

    print('The number of features after preprocessing is: ', len(df.columns) - 1)

    # Finding the colmuns having NaN values
    nan_features = df.columns[df.isna().any()].tolist()
    print("Features with NaN values:", nan_features)

    # Dropping any sample containing NaN or missing value
    df = df.dropna()

    # Replacing the infinite values with a very large number
    for column in df.columns:
        if (df[column] == np.inf).any():
            df[column].replace([np.inf], [np.finfo('float32').max], inplace=True)

    return df


def split_into_train_test(df):
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

    return X_train, X_test, y_train, y_test