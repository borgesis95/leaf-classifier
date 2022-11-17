



# -- Configuration parameters for csv's creation 

LABELS_PATHS = ['./labels.train.txt','./labels.test.txt']

## Training,Validation and test file tell where to save CSV
TRAINING_CSV_FILE = "csv/train.new_2.csv"
VALIDATION_CSV_FILE = "csv/validation.new_2.csv"
TEST_CSV_FILE="csv/test.new_2.csv"

# -- Configuration parameters for DataLoader creation

## Config information which say from which CSV create train,valid,test set
TRAINING_CSV_DATASET_FILE ="csv/train.new_2.csv"
VALIDATION_CSV_DATASET_FILE = "csv/validation.new_2.csv"
TEST_CSV_DATASET_FILE = "csv/test.new_2.csv"

BATCH_SIZE = 32
NUM_WORKERS = 3

## -- Main configuration

MODEL_NAME = "squeezenet"
EPOCHS = 1
LOAD_CHECKPOINT = True
LEARNING_RATE = 0.001
FEATURE_EXTR = False
DATA_AUGMENTATION = False

## -- Train configuration

CHECKPOINT_DIR = "checkpoint/checkpoint_13_11"
LOG_DIR = "logs/logs_13_11"