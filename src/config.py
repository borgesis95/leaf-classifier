
# -- extraction frames configuration
VIDEO_PATH = 'dataset/'

#Tell how many frames user want to extract from each video
FRAMES_PER_VIDEO = 150
#Describe where user want to save frame
FRAMES_FOLDER = "./frames/all/"

# If this feature is True frames for each video will be use for both train and test
DISTINCT_TRAINTEST_SET = False

TRAINING_RATE = 0.8

TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

#This array will be used only if DISTINCT_TRAINTEST_SET is True otherwise will be considered LABELS_ALL value
LABELS_DISTINCT_LIST = ['labels-2.train.txt','labels-2.test.txt']
LABELS_ALL = 'labels.all.txt'



# -- Configuration parameters for csv's creation 

LABELS_PATHS = ['labels.all.txt']
# LABELS_PATHS = ['./labels-2.train.txt','./labels-2.test.txt']
FRAMES_FOLDER_CSV = 'frames-2'

## Training,Validation and test file tell where to save CSV
TRAINING_CSV_FILE = "csv/train.all.csv"
VALIDATION_CSV_FILE = "csv/validation.all.csv"
TEST_CSV_FILE="csv/testing-2.all.csv"


# -- Configuration parameters for DataLoader creation

## Config information which say from which CSV create train,valid,test set
TRAINING_CSV_DATASET_FILE ="csv/train.set.csv"
VALIDATION_CSV_DATASET_FILE = "csv/validation.set.csv"
TEST_CSV_DATASET_FILE = "csv/test.set.csv"

BATCH_SIZE = 32
NUM_WORKERS = 3

## -- Main configuration

MODEL_NAME = "resnet"
EPOCHS = 25
LOAD_CHECKPOINT = True
LEARNING_RATE = 0.001
FEATURE_EXTR = True
DATA_AUGMENTATION = True

## -- Train configuration

CHECKPOINT_DIR = "checkpoint/epochs="+str(EPOCHS)
LOG_DIR = "logs/epochs="+str(EPOCHS)

## -- Gradio configuration
GRADIO_LOAD_CHECKPOINT_PATH="checkpoint/epochs="+str(EPOCHS)