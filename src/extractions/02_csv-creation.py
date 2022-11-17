import random 
import numpy as np
import pandas as pd
from src.utils.utils import split_train_val_test

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    label_path = "./labels.txt"
    # Loading labels files
    labelsFile = np.loadtxt(label_path,dtype=str, delimiter=",")

    
    images = []
    labels = []


    for i in range(len(labelsFile)):
        images.append('frames/' + labelsFile[i][0])
        labels.append(labelsFile[i][1])
    
    print("Total images: %d , Total Labels: %d" %(len(images),len(labels)))


# -- Creazione dataFrame --- 

datasetDataFrame = pd.DataFrame({
    'image' : images,
    'labels' :  labels
})

# Divisione del dataset in Train,Validation e Test set.
trainingDf , validationDf ,testDf = split_train_val_test(dataset=datasetDataFrame,percentual=[0.6,0.2,0.2])

trainingDf.to_csv('training.csv',index=None)
validationDf.to_csv('validation.csv',index=None)
testDf.to_csv('test.csv',index=None)
datasetDataFrame.to_csv('all.csv',index=None)
