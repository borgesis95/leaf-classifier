import random 
import numpy as np
import pandas as pd
from src.utils.utils import split_train_val_test
from src.config import TRAINING_CSV_FILE,VALIDATION_CSV_FILE,TEST_CSV_FILE,LABELS_PATHS,DISTINCT_TRAINTEST_SET,FRAMES_FOLDER_CSV
from sklearn.model_selection import train_test_split



def dataFrameCreation(label_path,type):
    images = []
    labels = []

    for i in range(len(label_path)):
        if DISTINCT_TRAINTEST_SET == True: images.append(FRAMES_FOLDER_CSV + '/'+ type +'/' +labelsFile[i][0])
        else: images.append(FRAMES_FOLDER_CSV + '/' +labelsFile[i][0])
        labels.append(labelsFile[i][1])
    print("Total images: %d , Total Labels: %d" %(len(images),len(labels)))

    datasetFrame = pd.DataFrame({
    'image' : images,
    'labels' :  labels
    })
    
    return datasetFrame






if __name__ == "__main__":
    random.seed(1395)
    np.random.seed(1359)

    images = []
    labels = []

    dataFrames = []
    for file in LABELS_PATHS:
        labelsFile = np.loadtxt(file,dtype=str, delimiter=",")
        dataset_type = file.split(".")[2]
        dataframe = dataFrameCreation(labelsFile,dataset_type)
        dataFrames.append(dataframe)






if(DISTINCT_TRAINTEST_SET == True):
    # split dataset on train,validation and test set.
    train,testval = train_test_split(dataFrames[0],test_size = 0)
    # Split second labels file into two partes for validation and test
    validation,test = train_test_split(dataFrames[1],test_size=0.4)
else:
    train,validation,test = split_train_val_test(dataset=dataFrames[0],percentual=[0.6,0.2,0.2])


train.to_csv(TRAINING_CSV_FILE,index=None)
validation.to_csv(VALIDATION_CSV_FILE,index=None)
test.to_csv(TEST_CSV_FILE,index=None)
