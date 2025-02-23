import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns
from pandas.plotting import table
#
#
# # from config import OUTPUT_FOLDER_PATH, TINYML_OUTPUT_FOLDER_PATH
# OUTPUT_FOLDER_PATH = r"D:\Projects\ev_charging_station\model_outputs"
# TINYML_OUTPUT_FOLDER_PATH = r"D:\Projects\ev_charging_station\tiny_ml_model_outputs"
#
# def plot_histogram(data, columnName, unit, title):
#     # Load the data
#     # data = pd.read_csv(dataName + '.csv')
#     n_samples = len(data)
#
#     # Define the font settings
#     calibri_font = {'family': 'Calibri',
#                     'color': 'black',
#                     'weight': 'normal',
#                     'size': 20,
#                     }
#     # Plot the histogram of the data
#     k = int(1 + 3.322 * math.log((data.shape[0]), 2))
#     data[columnName].hist(bins=k, density=False, alpha=1, edgecolor='black')  # label='Histogram'
#     plt.xlabel(f'{columnName} ({unit})', fontsize=50, labelpad=15, fontdict=calibri_font)
#     plt.ylabel('Density', fontsize=50, labelpad=15, fontdict=calibri_font)
#     plt.title(title, fontsize=50, pad=22, fontdict=calibri_font)
#
#     # Calculate the mean and standard deviation of the data
#     mu, std = data[columnName].mean(), data[columnName].std()
#
#     # Plot the normal curve
#     xmin, xmax = plt.xlim()
#     bin_width = (xmax - xmin) / k
#     x = np.linspace(xmin - 3 * std, xmax + 3 * std, 1000)
#     p = stats.norm.pdf(x, mu, std) * n_samples * bin_width
#     plt.plot(x, p, linewidth=3, color='#993d4d')  # label='Normal Distribution'
#
#     minumum = data.min().min()
#     maximum = data.max().max()
#
#     # Create custom legend labels
#     legend_labels = [
#         f'Min:{minumum:.2f}',
#         f'Max: {maximum:.2f}',
#         f'Mean: {mu:.2f}',
#         f'Std: {std:.2f}',
#         f'Number of Samples: {n_samples}',
#     ]
#
#     # Create a custom legend using dummy plot objects with no lines or markers
#     for label in legend_labels:
#         plt.plot([], [], ' ', label=label)
#
#     # Display the legend with specified font size
#     plt.legend(fontsize=35)
#
#     plt.xticks(fontsize=36, fontname="Calibri")
#     plt.yticks(fontsize=36, fontname="Calibri")
#
#     plt.rcParams['font.family'] = 'Calibri'
#     plt.show()
#
#     # alternatively you can directly save the figure as: plt.savefig(filename), then plt.close()
#
#
# # plot_histogram(pd.read_csv(OUTPUT_FOLDER_PATH+'/per_sample_memory.csv'), 'Memory' , 'KB', 'ML_MLP Memory Consumption')
# plot_histogram(pd.read_csv(TINYML_OUTPUT_FOLDER_PATH+'/per_sample_memory.csv'), 'Memory', 'Bytes', 'TinyML_MLP Memory Consumption')
#
# plot_histogram(pd.read_csv(OUTPUT_FOLDER_PATH+'/prediction_times.csv'), 'Time', 'ms', 'ML_MLP Inference Time')
# plot_histogram(pd.read_csv(TINYML_OUTPUT_FOLDER_PATH+'/prediction_times.csv'), 'Time','\u03BC' + 's', 'TinyML_MLP Inference Time')
#
#
#
#


# Load the CSV results for both models
mlp_results = pd.read_csv('D:\Projects\ev_charging_station\model_outputs\k_fold_results_mlp.csv')
rf_results = pd.read_csv('D:\Projects\ev_charging_station\model_outputs\k_fold_results.csv')
# Create a comparative DataFrame
comparison_df = pd.DataFrame({
    'Fold': mlp_results['Fold'].round(6),
    'MLP Accuracy': mlp_results['Accuracy'].round(6),
    'RF Accuracy': rf_results['Accuracy'].round(6),
    'MLP Precision': mlp_results['Precision'].round(6),
    'RF Precision': rf_results['Precision'].round(6),
    'MLP Recall': mlp_results['Recall'].round(6),
    'RF Recall': rf_results['Recall'].round(6),
    'MLP F1 Score': mlp_results['F1 Score'].round(6),
    'RF F1 Score': rf_results['F1 Score'].round(6)
})

# For making comparisons images

# Define metrics for comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
mlp_columns = [f'MLP {metric}' for metric in metrics]
rf_columns = [f'RF {metric}' for metric in metrics]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()  # Flatten axes array for easy iteration

for i, metric in enumerate(metrics):
    sns.lineplot(x='Fold', y=f'MLP {metric}', data=comparison_df, ax=axes[i], label='MLP', marker='o')
    sns.lineplot(x='Fold', y=f'RF {metric}', data=comparison_df, ax=axes[i], label='Random Forest', marker='o')
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel('Fold')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_metrics.png')
plt.show()



# create different comparison tables
accuracy_df = pd.DataFrame({
    'Fold': mlp_results['Fold'],
    'MLP Accuracy': mlp_results['Accuracy'].round(6),
    'RF Accuracy': rf_results['Accuracy'].round(6)
})

precision_df = pd.DataFrame({
    'Fold': mlp_results['Fold'],
    'MLP Precision': mlp_results['Precision'].round(6),
    'RF Precision': rf_results['Precision'].round(6)
})

recall_df = pd.DataFrame({
    'Fold': mlp_results['Fold'],
    'MLP Recall': mlp_results['Recall'].round(6),
    'RF Recall': rf_results['Recall'].round(6)
})

f1_df = pd.DataFrame({
    'Fold': mlp_results['Fold'],
    'MLP F1 Score': mlp_results['F1 Score'].round(6),
    'RF F1 Score': rf_results['F1 Score'].round(6)
})

# Create a figure with subplots for each metric
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()  # Flatten the 2D array to 1D

# List of DataFrames and titles for each subplot
dfs = [accuracy_df, precision_df, recall_df, f1_df]
titles = ['Accuracy Comparison', 'Precision Comparison', 'Recall Comparison', 'F1 Score Comparison']

# Create tables in each subplot
for ax, df, title in zip(axs, dfs, titles):
    ax.axis('off')  # Hide the axes
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.1] * len(df.columns))
    tbl.auto_set_font_size(False)  # Activate auto font size
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)  # Scale table
    ax.set_title(title, fontsize=14)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_metrics_tables.png', bbox_inches='tight')
plt.show()