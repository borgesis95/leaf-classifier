from torch_lr_finder import LRFinder
from models import AlexNet,ResNet,vgg16,GoogleNet
from torch.optim import SGD
from torch import nn
from utils import load_dataset

def get_alexNet_LR():

    start_lr = 0.0002 # Get From https://pypi.org/project/torch-lr-finder/
    end_lr = 0.2

    model = AlexNet().get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),start_lr,momentum=0.99)

    train_loader,validation_loader,test_loader =  load_dataset()

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader=train_loader,val_loader=validation_loader,end_lr=end_lr, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

def get_googleNet_LR():
    
    start_lr = 0.0002 # Get From https://pypi.org/project/torch-lr-finder/
    end_lr = 0.2

    model = GoogleNet().get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),start_lr,momentum=0.99)

    train_loader,validation_loader,test_loader =  load_dataset()

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader=train_loader,val_loader=validation_loader,end_lr=end_lr, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    # get_alexNet_LR()     #0.00283 ALEXNET
    get_googleNet_LR()  # 0.000866 GoogleNet
