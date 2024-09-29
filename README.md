# On TinyML and Cybersecurity: Electric Vehicle Charging Infrastructure Use Case

This repository contains the code implementation related to the research paper titled "On TinyML and Cybersecurity: Electric Vehicle Charging Infrastructure Use Case" by Fatemeh Dehrouyeh, Li Yang, Firouz Badrkhani Ajaei, and Abdallah Shami.

## Overview

This project aims to explore the integration of Tiny Machine Learning (TinyML) and cybersecurity measures within the context of Electric Vehicle (EV) charging infrastructure. The objective is to enhance the security and efficiency of EV charging systems using lightweight machine learning models.

## Features

- **Data Acquisition**: Collect and preprocess data related to EV charging behavior.
- **Model Implementation**: Implement TinyML models to predict and detect anomalies in charging patterns.
- **Cybersecurity Measures**: Apply cybersecurity protocols to secure data and communication in the charging infrastructure.
- **Evaluation Metrics**: Assess the performance of models with relevant metrics.

## Requirements

- Python 3.x
- Required Libraries:
    - numpy
    - pandas
    - scikit-learn
    - TensorFlow 
    - more requirements are listed under the file requirements.txt

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:sunilthapa43/ev_charging_station.git
   ```

2. Set up .env file and define output folder paths
   ```bash
   # sample .env contents
   OUTPUT_FOLDER_PATH=YOUR_DESIRED_OUTPUT_FOLDER_PATH
   PLOTS_AND_FIGURES_PATH=YOUR_DESIRED_OUTPUT_FOLDER_PATH
   ```
4. Create a virtual environment 
   ```bash
   virtualenv .venv
   ```
4. Install the requirements under the virtual environment
   ```bash
   pip install -r requirements.txt
   ```
5. Run the main file
   ```bash
   python3 main.py
   ```
