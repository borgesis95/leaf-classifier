
# -- CONFIGURATION FOR EXTRACT FRAMES (01_frames.py) ---

#Tell the folder where you put the videos from which you want to extract frames
VIDEO_PATH = 'dataset/'

#How many frames user want to extract from each video
FRAMES_PER_VIDEO = 150

#Describe where user want to save frames just extracted
FRAMES_FOLDER = "./frames/all/"

# If this feature is True frames for each video will be use for both train and test
DISTINCT_TRAINTEST_SET = False

#Percentage of video that will be used for training 
TRAINING_RATE = 0.8

#Where DISTINCT_TRAINTEST_SET = TRUE This two variables ell where to put images (es. frames/train, frames/all) if DISTINCT_TRIANTEST_SET 
# is equal to false, frames will be put on main folder(es frames/)
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

#This array will be used only if DISTINCT_TRAINTEST_SET is True otherwise will be considered LABELS_ALL value
LABELS_DISTINCT_LIST = ['labels.train.txt','labels.test.txt']

# This will be used only if DISINTICT_TRAINTEST_SET = False
LABELS_ALL = 'labels.all.txt'



# -- CONFIGURATION FOR CSV CREATION (02_csv.py) --

# Tells from which file read .txt files generated in the previous step
LABELS_PATHS = ['labels.all.txt']

# LABELS_PATHS = ['./labels-2.train.txt','./labels-2.test.txt']
FRAMES_FOLDER_CSV = 'frames/all'

## Training,Validation and test file tell where to save CSV
TRAINING_CSV_FILE = "csv/train.all.csv"
VALIDATION_CSV_FILE = "csv/validation.all.csv"
TEST_CSV_FILE="csv/test.all.csv"


# -- CONFIGURATION PARAMETERS FOR MAIN(main.py) ---

# This 3 variables tells which csv use for dataLoader
TRAINING_CSV_DATASET_FILE ="csv/train.set.csv"
VALIDATION_CSV_DATASET_FILE = "csv/validation.set.csv"
TEST_CSV_DATASET_FILE = "csv/test.set.csv"

BATCH_SIZE = 32
NUM_WORKERS = 3

MODEL_NAME = "alexnet"
EPOCHS = 50
# Tells if retrieve previous checkpoint saved from other iterations
LOAD_CHECKPOINT = True
LEARNING_RATE = 0.001
FEATURE_EXTR = True
DATA_AUGMENTATION = True

## -- Train configuration

#Folders where to save checkpoints/logs created
CHECKPOINT_DIR = "checkpoint/epochs="+str(EPOCHS)
LOG_DIR = "logs/epochs="+str(EPOCHS)

# --- GRADIO ---
#Tells from which folder load checkpoints 
GRADIO_LOAD_CHECKPOINT_PATH="checkpoint/epochs="+str(EPOCHS)