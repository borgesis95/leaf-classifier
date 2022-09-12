# default mean and std needed by pretrained models from pytorch
from Testing import test_classifier
from models import get_Squeezenet_model
from Utils import load_dataset
from Trainer import trainval_classifier
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16

from torch import nn
from sklearn.metrics import accuracy_score



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def training(modelName,isDAactive, isPretrainedActive,epochsNumber):

    print("--- Modello utilizzato --> ", modelName)
    
    train_loader,validation_loader,test_loader = load_dataset(False)

    # Ci saranno tre classi presenti
    classesNumber = 4
    if modelName =='SqueezeNet':
          model = squeezenet1_0(pretrained=True)
          model.classifier[1] = nn.Conv2d(512, classesNumber, kernel_size=(1, 1), stride=(1, 1))
          model.num_classes = classesNumber
    
    else:
        print("VGG16")
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, classesNumber)

    #Allenamento
    model = trainval_classifier(model, isPretrainedActive, modelName, train_loader, validation_loader, lr=0.001, exp_name=modelName, momentum=0.99, epochs=epochsNumber)

    #test (viene passato il modello appena allenato)
    predictions,labels = test_classifier(model,test_loader)
    accuracy = accuracy_score(labels,predictions)*100

    print("Accuracy of : ",modelName," : ",accuracy)

    return accuracy,model



if  __name__ =='__main__':
     training('VGG16',False,False,10)         

