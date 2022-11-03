import os
import random 
import numpy as np
import pandas as pd
from script.utils.utils import split_train_val_test


def  put_on_labels_txt():
    class_dictionary = {
        "Limone": 0,
        "Melograno":1,
        "Mango":2
    }

    source = ['Limone','Melograno','Mango']
    frames_path = './external_dataset/'

    labels = open ('./labels_2.txt','a')

    for curr_folder in (source):
        path =  frames_path + curr_folder
        print("path",path)
        i = 0
        for filename in os.listdir(path):
            print("i",i)
            print("source:",curr_folder)
            print("frame",filename)
            label_class = class_dictionary[curr_folder]
            labels.write(curr_folder+'/'+filename+ ', %d\n' %(label_class))
    labels.close()


def extraction_csv():
    random.seed(1234)
    np.random.seed(1234)
    labelsFile = np.loadtxt("./labels_2.txt",dtype=str, delimiter=",")

    images = []
    labels = []
    for i in range(len(labelsFile)):
        images.append('external_dataset/'+labelsFile[i][0])
        labels.append(labelsFile[i][1])
    print("Total images: %d , Total Labels: %d" %(len(images),len(labels)))
    datasetDataFrame = pd.DataFrame({
        'image' : images,
        'labels' :  labels
    })
    trainingDf , validationDf ,testDf = split_train_val_test(dataset=datasetDataFrame,percentual=[0.6,0.2,0.2])
    trainingDf.to_csv('training_2.csv',index=None)
    validationDf.to_csv('validation_2.csv',index=None)
    datasetDataFrame.to_csv('all.csv_2',index=None)




if __name__ == "__main__":

    put_on_labels_txt()
    extraction_csv()
    

     
    
