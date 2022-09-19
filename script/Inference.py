from models import GoogleNet,AlexNet
import torchvision.transforms as transforms
from PIL import Image
import torch
import os


def get_transform(im):
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.511, 0.511, 0.511], [0.0947, 0.0947, 0.0948])
    ])

    return train_transform(im)


def inference(modelname,filename):
    model = AlexNet().get_model()
    #Carichiamo il dizionario 
    
   
    lr = 0.001
    if(os.path.isfile('./checkpoint/' + modelname +'_lr=' +str(lr)+ '_checkpoint.pth')):
        model.load_state_dict(torch.load('./checkpoint/'+ modelname +'_lr=' +str(lr)+'_checkpoint.pth')['state_dict'])   

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



