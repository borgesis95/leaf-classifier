



# -- Configuration parameters for csv's creation 

LABELS_PATHS = ['./labels.train.txt','./labels.test.txt']

## Training,Validation and test file tell where to save CSV
TRAINING_CSV_FILE = "csv/train.new_4.csv"
VALIDATION_CSV_FILE = "csv/validation.new_4.csv"
TEST_CSV_FILE="csv/test.new_4.csv"

# -- Configuration parameters for DataLoader creation

## Config information which say from which CSV create train,valid,test set
TRAINING_CSV_DATASET_FILE ="csv/train.new_3.csv"
VALIDATION_CSV_DATASET_FILE = "csv/validation.new_3.csv"
TEST_CSV_DATASET_FILE = "csv/test.new_3.csv"

BATCH_SIZE = 32
NUM_WORKERS = 3

## -- Main configuration

MODEL_NAME = "squeezenet"
EPOCHS = 25
LOAD_CHECKPOINT = True
LEARNING_RATE = 0.0005
FEATURE_EXTR = False
DATA_AUGMENTATION = True

## -- Train configuration

CHECKPOINT_DIR = "checkpoint/epochs="+str(EPOCHS)
LOG_DIR = "logs/epochs="+str(EPOCHS)

## -- Gradio configuration
GRADIO_LOAD_CHECKPOINT_PATH="checkpoint/epochs="+str(EPOCHS)