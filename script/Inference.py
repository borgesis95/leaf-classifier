from re import I
from script.models import GoogleNet, ResNet, SqueezeNet
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
from torch import nn

from torchvision.models import squeezenet1_1,AlexNet,vgg16,resnet18
from torchvision.models.googlenet import googlenet

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

    if modelname == "googlenet":
        model = googlenet(pretrained=True)
        num_class = 3
        model.fc = nn.Linear(1024, num_class)

    #Carichiamo il dizionario 
    if(os.path.isfile('./checkpoint/'+ param_file)):
            print("carico i parametri..")
            model.load_state_dict(torch.load('./checkpoint/'+ param_file)['state_dict'])
   
    
    # if(os.path.isfile('./checkpoint/' + modelname +'_lr=' +str(lr)+'_fe='+str(feature_extraction)+ '_checkpoint.pth')):
    #     print("carico i parametri..")
    #     model.load_state_dict(torch.load('./checkpoint/' + modelname + '_lr=' + str(lr)+'_fe='+str(feature_extraction)+ '_checkpoint.pth')['state_dict'])   
    
    return model


# def inference(modelname,filename):

#     model = ""

#     if modelname =="alexnet":
#         print("AlexNet")
#         model = AlexNet()
#         model.classifier[6] = nn.Linear(4096, 3)

#     if modelname =="resnet":
#         model = resnet18()
#         model.fc = nn.Linear(512, 3)
#         model.num_classes = 3

#     if modelname == "googlenet":
#         print("GOOGLENET")
#         model = googlenet(pretrained=True)
#         num_class = 3
#         model.fc = nn.Linear(1024, num_class)

#     #Carichiamo il dizionario 
    
   
#     lr = 0.001
#     if(os.path.isfile('./checkpoint/' + modelname +'_lr=' +str(lr)+ '_checkpoint.pth')):
#         print("carico i parametri..")
#         model.load_state_dict(torch.load('./checkpoint/' + modelname + '_lr=' + str(lr)+'_checkpoint.pth')['state_dict'])   

#     # Deve essere eseguito il preprocessing
#     #DAFARE

#     image = Image.open(filename)
#     image = get_transform(image)

#     batch_t = torch.unsqueeze(image, 0)
#     model.eval()
#     out_predict = model(batch_t)
#     _,index = torch.max(out_predict,1)
#     labels = ['Alloro','Edera','Nespola']

#     print("Index",index)
#     print("LABEL",labels[index[0]])
#     return labels[index[0]]



def predict(data,parameters):


    print("input",parameters)
    labels = ['Alloro','Edera','Nespola']
    image = transforms.Compose([
        transforms.Resize(320),
        # wastebin will be centered in the photo (photo from smartphone has usually height > width)
        # use centerCrop to exclude other parts of images not necessary, like floor
        # tried also witouth centerCrop, using centerCrop gives better result
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])(Image.fromarray(data.astype('uint8'), 'RGB')).unsqueeze(0)

    
    model = get_model(param_file=parameters)
    with torch.no_grad():
        #  prediction = model(image)[0]
         prediction = torch.nn.functional.softmax(model(image)[0], dim=0)

         confidences = {labels[i]: float(prediction[i]) for i in range(3)}    
    return confidences


