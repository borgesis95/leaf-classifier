from re import I
from script.models import GoogleNet,AlexNet, ResNet, SqueezeNet
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
from torch import nn

from torchvision.models import squeezenet1_1,AlexNet as AlNet,vgg16,resnet18
from torchvision.models.googlenet import googlenet

def get_transform(im):
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
    ])

    return train_transform(im)


def inference(modelname,filename):

    model = ""
    if modelname =="resnet":
        model = resnet18()
        model.fc = nn.Linear(512, 3)
        model.num_classes = 3

    if modelname == "googlenet":
        print("GOOGLENET")
        model = googlenet(pretrained=True)
        num_class = 3
        model.fc = nn.Linear(1024, num_class)

    #Carichiamo il dizionario 
    
   
    lr = 0.001
    if(os.path.isfile('./checkpoint/' + modelname +'_lr=' +str(lr)+ '_checkpoint.pth')):
        print("carico i parametri..")
        model.load_state_dict(torch.load('./checkpoint/' + modelname + '_lr=' + str(lr)+'_checkpoint.pth')['state_dict'])   

    # Deve essere eseguito il preprocessing
    #DAFARE

    image = Image.open(filename)
    image = get_transform(image)

    batch_t = torch.unsqueeze(image, 0)
    model.eval()
    out_predict = model(batch_t)
    _,index = torch.max(out_predict,1)
    labels = ['Alloro','Mandarino','Ulivo','Non saprei']

    print("Index",index)
    print("LABEL",labels[index[0]])
    return labels[index[0]]



