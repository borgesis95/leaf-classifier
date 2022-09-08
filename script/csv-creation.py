import random 
import numpy as np
import pandas as pd

from utils import split_train_val_test

if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)

    # Caricamento del file labels.txt
    labelsFile = np.loadtxt("./labels.txt",dtype=str, delimiter=",")

    # Dal file TXT verrranno suddivise le immagini e la classe associata ad essa
    images = []
    labels = []

    for i in range(len(labelsFile)):

        # Vengono inserite nell'array i path delle immagini salvate sul file labels.txt
        images.append('frame/' + labelsFile[i][0])
        #Vengono inserite le classi associate
        labels.append(labelsFile[i][1])
    
    print("Total images: %d , Total Labels: %d" %(len(images),len(labels)))


# -- Creazione dataFrame --- 

datasetDataFrame = pd.DataFrame({
    'image' : images,
    'labels' :  labels
})

# Divisione del dataset in Train,Validation e Test set.
trainingDf , validationDf ,testDf = split_train_val_test(dataset=datasetDataFrame,percentual=[0.5,0.2,0.3])

trainingDf.to_csv('training.csv',index=None)
validationDf.to_csv('validation.csv',index=None)
datasetDataFrame.to_csv('all.csv',index=None)

ids, classes = zip(*{
        0: "Alloro",
        1: "Mandarino",
        2: "Ulivo"
    }.items())
ids = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
ids.to_csv('dataset/classes.csv')

