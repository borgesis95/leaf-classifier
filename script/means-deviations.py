from CsvImageDataset import CSVImageDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from matplotlib import pyplot as plt


datasetData = CSVImageDataset('./all.csv')


print ('Dataset Length: % d' %(datasetData.__len__()))


means = np.zeros(3)
stdevs = np.zeros(3)

for data in datasetData:
       img = data[0]
       for i in range(3):
            img = np.asarray(img)
            means[i] += img[i, :, :].mean()
            stdevs[i] += img[i, :, :].std()
means = np.asarray(means) / datasetData.__len__()
stdevs = np.asarray(stdevs) / datasetData.__len__()


print("{} : normMean = {}".format(type, means))
print("{} : normstdevs = {}".format(type, stdevs))
