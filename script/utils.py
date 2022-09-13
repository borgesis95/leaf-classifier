from os.path import join
from numpy.core.records import array
from pandas.core.frame import DataFrame

from torchvision import transforms
from sklearn.model_selection import train_test_split
from CsvImageDataset import CSVImageDataset
from torch.utils.data import DataLoader


# Il dataset viene diviso prima in due parti training e test.
# Il testset a sua volta viene suddiviso in  validation e testset
def split_train_val_test(dataset: DataFrame,percentual: array):
    train,testval = train_test_split(dataset,test_size = percentual[1] + percentual[2])
    val,test = train_test_split(testval,test_size = percentual[2]/(percentual[1]+percentual[2]))
    return train,val,test


def load_dataset(isDataAugmentationActive = False):

    if isDataAugmentationActive:
        train_transform = transforms.Compose([
        transforms.RandomAffine((-20,20),shear=[-20,20]), 
        transforms.RandomRotation((-40,40)),
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
        ])

    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
        ])
                        
                        
    

    test_transform = transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # -- Caricamento dai CSV ---
    train_dataset = CSVImageDataset('./training.csv',transform=train_transform)
    validation_dataset = CSVImageDataset('./validation.csv',transform=train_transform)
    test_dataset = CSVImageDataset('./validation.csv',transform=train_transform)

    # -- Utilizzo Dataloader ---

    batch_size = 32
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader,valid_loader,test_loader