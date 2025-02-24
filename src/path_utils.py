import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DATASET_FILE = config['PATHS']['DATASET_FILE']
DESCRIPTOR_FILE = config['PATHS']['DESCRIPTOR_FILE']
SOURCES_FILE = config['PATHS']['SOURCES_FILE']

MODEL_SAVE_PATH = config['PATHS']['MODEL_SAVE_PATH']
RESULTS_PATH = config['PATHS']['RESULTS_PATH']
FIGURES_PATH = config['PATHS']['FIGURES_PATH']

CSV_TRANSACTIONS_PATH = config['PATHS']['CSV_TRANSACTIONS_PATH']
JSON_TRANSACTIONS_PATH = config['PATHS']['JSON_TRANSACTIONS_PATH']