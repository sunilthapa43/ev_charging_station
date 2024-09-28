import pandas as pd
from .constants import keep_features, DoS_Types, Brute_Force_Types, Web_Attack_types, Others
import os
from config import DATASET_FOLDER_PATH

file_root = DATASET_FOLDER_PATH

# Load the dataset
df1 = pd.read_csv(file_root + '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df2 = pd.read_csv(file_root + '/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df3 = pd.read_csv(file_root + '/Friday-WorkingHours-Morning.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df4 = pd.read_csv(file_root + '/Monday-WorkingHours.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df5 = pd.read_csv(file_root + '/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df6 = pd.read_csv(file_root + '/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df7 = pd.read_csv(file_root + '/Tuesday-WorkingHours.pcap_ISCX.csv', encoding='latin1', low_memory=False)
df8 = pd.read_csv(file_root + '/Wednesday-workingHours.pcap_ISCX.csv', encoding='latin1', low_memory=False)

#Get the features
column_names1 = df1.columns.tolist()
print('The features are: ', column_names1)
print('The number of features is: ', len(column_names1))

## Creating Our DataFrame
# Merge the DataFrames horizontally based on common columns (all columns)
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=0)
combined_df.head()

# Remove spaces before the first letter of column names
combined_df = combined_df.rename(columns=lambda x: x.strip())
print('The number of rows in the combined dataset is: ', combined_df.shape[0])

#The features that are related to Electric Vehicle Charging Stations are kept and the other features are removed
# Remove the features not in 'keep_features'
Final_df = combined_df.drop(columns=[col for col in combined_df.columns if col not in keep_features])
final_column_names = Final_df.columns.tolist()
print('number of selected features: ', len(final_column_names)-1)

#Getting the categories of the Labels
print('Labels are: ', Final_df['Label'].unique().tolist())
print('The number of categories is: ', Final_df['Label'].nunique())

# Count the occurrences of each label
value_counts = Final_df['Label'].value_counts()

# Calculate the total number of samples
total_number_of_samples = len(Final_df)

# Calculate the percentage
percentages = (value_counts / total_number_of_samples) * 100

# Create a DataFrame to display both counts and percentages
Distribution_Results = pd.DataFrame({'Value Counts':value_counts,'Percentages':percentages})
print(Distribution_Results)



Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Dos' if x in DoS_Types else x)
Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Brute_Force' if x in Brute_Force_Types else x)
Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Web_Attack' if x in Web_Attack_types else x)
Final_df['Label'] = Final_df['Label'].apply(lambda x: 'Bot/Infiltration/Heartbleed' if x in Others else x)


def get_final_df():
    return Final_df