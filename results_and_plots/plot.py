import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


# from config import OUTPUT_FOLDER_PATH, TINYML_OUTPUT_FOLDER_PATH
OUTPUT_FOLDER_PATH = r"D:\Projects\ev_charging_station\model_outputs"
TINYML_OUTPUT_FOLDER_PATH = r"D:\Projects\ev_charging_station\tiny_ml_model_outputs"

def plot_histogram(data, columnName, unit, title):
    # Load the data
    # data = pd.read_csv(dataName + '.csv')
    n_samples = len(data)

    # Define the font settings
    calibri_font = {'family': 'Calibri',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 20,
                    }
    # Plot the histogram of the data
    k = int(1 + 3.322 * math.log((data.shape[0]), 2))
    data[columnName].hist(bins=k, density=False, alpha=1, edgecolor='black')  # label='Histogram'
    plt.xlabel(f'{columnName} ({unit})', fontsize=50, labelpad=15, fontdict=calibri_font)
    plt.ylabel('Density', fontsize=50, labelpad=15, fontdict=calibri_font)
    plt.title(title, fontsize=50, pad=22, fontdict=calibri_font)

    # Calculate the mean and standard deviation of the data
    mu, std = data[columnName].mean(), data[columnName].std()

    # Plot the normal curve
    xmin, xmax = plt.xlim()
    bin_width = (xmax - xmin) / k
    x = np.linspace(xmin - 3 * std, xmax + 3 * std, 1000)
    p = stats.norm.pdf(x, mu, std) * n_samples * bin_width
    plt.plot(x, p, linewidth=3, color='#993d4d')  # label='Normal Distribution'

    minumum = data.min().min()
    maximum = data.max().max()

    # Create custom legend labels
    legend_labels = [
        f'Min:{minumum:.2f}',
        f'Max: {maximum:.2f}',
        f'Mean: {mu:.2f}',
        f'Std: {std:.2f}',
        f'Number of Samples: {n_samples}',
    ]

    # Create a custom legend using dummy plot objects with no lines or markers
    for label in legend_labels:
        plt.plot([], [], ' ', label=label)

    # Display the legend with specified font size
    plt.legend(fontsize=35)

    plt.xticks(fontsize=36, fontname="Calibri")
    plt.yticks(fontsize=36, fontname="Calibri")

    plt.rcParams['font.family'] = 'Calibri'
    plt.show()


# plot_histogram(pd.read_csv(OUTPUT_FOLDER_PATH+'/per_sample_memory.csv'), 'Memory' , 'KB', 'ML_MLP Memory Consumption')
plot_histogram(pd.read_csv(TINYML_OUTPUT_FOLDER_PATH+'/per_sample_memory.csv'), 'Memory', 'Bytes', 'TinyML_MLP Memory Consumption')

plot_histogram(pd.read_csv(OUTPUT_FOLDER_PATH+'/prediction_times.csv'), 'Time', 'ms', 'ML_MLP Inference Time')
plot_histogram(pd.read_csv(TINYML_OUTPUT_FOLDER_PATH+'/prediction_times.csv'), 'Time','\u03BC' + 's', 'TinyML_MLP Inference Time')
