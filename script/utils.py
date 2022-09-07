from os.path import join
from numpy.core.records import array
from pandas.core.frame import DataFrame

from torchvision import transforms
from sklearn.model_selection import train_test_split



# Il dataset viene diviso prima in due parti training e test.
# Il testset a sua volta viene suddiviso in  validation e testset
def split_train_val_test(dataset: DataFrame,percentual: array):
    train,testval = train_test_split(dataset,test_size = percentual[1] + percentual[2])
    val,test = train_test_split(testval,test_size = percentual[2]/(percentual[1]+percentual[2]))
    return train,val,test