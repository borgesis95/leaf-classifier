from os.path import join
from numpy.core.records import array
from pandas.core.frame import DataFrame

from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.CsvImageDataset import CSVImageDataset
from torch.utils.data import DataLoader
from src.config import TRAINING_CSV_DATASET_FILE,TEST_CSV_DATASET_FILE,VALIDATION_CSV_DATASET_FILE,BATCH_SIZE,NUM_WORKERS


def split_train_val_test(dataset: DataFrame,percentual: array):
    train,testval = train_test_split(dataset,test_size = percentual[1] + percentual[2])
    val,test = train_test_split(testval,test_size = percentual[2]/(percentual[1]+percentual[2]))
    return train,val,test


def load_dataset(data_augmentation):
    if data_augmentation:
        train_transforms = transforms.Compose([
                        #    transforms.Resize(224),
                           transforms.RandomCrop(224),
                           transforms.RandomVerticalFlip(),
                        #    transforms.RandomRotation(180),
                           transforms.ColorJitter(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    else:
        train_transforms = transforms.Compose([
                            # transforms.Resize(224),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

    
                        
    validation_transforms = transforms.Compose([
                            # transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = CSVImageDataset(TRAINING_CSV_DATASET_FILE,transform=train_transforms)
    validation_dataset = CSVImageDataset(VALIDATION_CSV_DATASET_FILE,transform=validation_transforms)
    test_dataset = CSVImageDataset(TEST_CSV_DATASET_FILE,transform=validation_transforms)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)




    return train_loader,valid_loader,test_loader,validation_dataset



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def write_reacaps(saving_text):
    
    with open('recaps.txt', 'a') as f:
        f.write(saving_text +'\n')