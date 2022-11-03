from re import I
# from script.models import 
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
from torch import nn

from torchvision.models import squeezenet1_1,AlexNet,vgg16,resnet18
from matplotlib import pyplot as plt

def get_transform(im):
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
    ])

    return train_transform(im)


def get_model(param_file):
    model = ""

    modelname  = param_file.split('_')[0]

    print("modelname",modelname)

    if modelname =="alexnet":
        model = AlexNet()
        model.classifier[6] = nn.Linear(4096, 3)

    if modelname =="resnet":
        model = resnet18()
        model.fc = nn.Linear(512, 3)
        model.num_classes = 3

    if modelname == "squeezenet":
        model = squeezenet1_1()
        num_class = 3
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_class


    print("paramfile"+ param_file)
    #Load dictionary
    if(os.path.isfile('./checkpoint/'+ param_file)):
            print("carico i parametri..")
            model.load_state_dict(torch.load('./checkpoint/'+ param_file)['state_dict'])    
    return model




def predict(data,parameters):


    print("input",parameters)
    # labels = ['Limone','melograno','mango']
    labels = ['Alloro','Edera','Nespola']
    image = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])(Image.fromarray(data.astype('uint8'), 'RGB')).unsqueeze(0)
  


    
    model = get_model(param_file=parameters)
    with torch.no_grad():
        
         prediction = torch.nn.functional.softmax(model(image)[0], dim=0)

         confidences = {labels[i]: float(prediction[i]) for i in range(3)}    
    return confidences


