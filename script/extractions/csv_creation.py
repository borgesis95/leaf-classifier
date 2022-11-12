import random 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def createDataFrame(label_file):
    label_path = './'+label_file
    myFile = np.loadtxt(label_path,dtype=str,delimiter=",")

    file_type = label_file.split('.')[1]

    images = []
    labels = []

    for i in range(len(myFile)):
        images.append('frames/'+file_type+'/' + myFile[i][0])
        labels.append(myFile[i][1])
    
    print("Total images: %d , Total Labels: %d" %(len(images),len(labels)))
    return pd.DataFrame({
    'image' : images,
    'labels' :  labels
})



if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    label_path = "./labels.txt"

    labels_files = ['labels.train.txt','labels.test.txt'];
    # Loading labels files
    dataframes = []

    for file in labels_files:
        dataframe = createDataFrame(file)
        dataframes.append(dataframe)


    trainingDf , val  = train_test_split(dataframes[0],test_size = 0)
    validationDf,testDf = train_test_split(dataframes[1],test_size = 0.5)

    trainingDf.to_csv('train.new.csv',index=None)
    validationDf.to_csv('validation.new.csv',index=None)
    testDf.to_csv('test.new.csv',index=None)
