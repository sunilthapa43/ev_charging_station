import os
from dotenv import load_dotenv

# load dotenv
load_dotenv()

OUTPUT_FOLDER_PATH = os.getenv('OUTPUT_FOLDER_PATH')
DATASET_FOLDER_PATH = os.getenv('DATASET_FOLDER_PATH')
TINYML_OUTPUT_FOLDER_PATH = os.getenv('TINYML_OUTPUT_FOLDER_PATH')