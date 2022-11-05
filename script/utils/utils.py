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


def load_dataset(data_augmentation,training_csv ='training',validation_csv='validation',test_csv='test'):
    if data_augmentation:
        train_transforms = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(180),
                           transforms.RandomCrop(170),
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
                            # transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = CSVImageDataset('./'+training_csv+'.csv',transform=train_transforms)
    validation_dataset = CSVImageDataset('./'+validation_csv+'.csv',transform=validation_transforms)
    test_dataset = CSVImageDataset('./'+test_csv+'.csv',transform=validation_transforms)

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


def write_reacaps(saving_text):
    
    with open('recaps.txt', 'a') as f:
        f.write(saving_text +'\n')