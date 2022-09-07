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
    images = [+]