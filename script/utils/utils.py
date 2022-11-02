from os.path import join
from numpy.core.records import array
from pandas.core.frame import DataFrame

from torchvision import transforms
from sklearn.model_selection import train_test_split
from script.CsvImageDataset import CSVImageDataset
from torch.utils.data import DataLoader


# Il dataset viene diviso prima in due parti training e test.
# Il testset a sua volta viene suddiviso in  validation e testset
def split_train_val_test(dataset: DataFrame,percentual: array):
    train,testval = train_test_split(dataset,test_size = percentual[1] + percentual[2])
    val,test = train_test_split(testval,test_size = percentual[2]/(percentual[1]+percentual[2]))
    return train,val,test


def load_dataset(data_augmentation):


    if data_augmentation:
        print("Dataaugmentation")
        train_transforms = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.ColorJitter(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:

        train_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

    
                        
    validation_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # -- Caricamento dai CSV ---
    train_dataset = CSVImageDataset('./training.csv',transform=train_transforms)
    validation_dataset = CSVImageDataset('./validation.csv',transform=validation_transforms)
    test_dataset = CSVImageDataset('./validation.csv',transform=validation_transforms)

    # -- Utilizzo Dataloader ---

    batch_size = 32
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader,valid_loader,test_loader,train_dataset



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

