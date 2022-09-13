# default mean and std needed by pretrained models from pytorch
from Testing import test_classifier
from Utils import load_dataset
from Trainer import trainval_classifier
from torch import nn
from sklearn.metrics import accuracy_score
from Models import SqueezeNet,AlexNet,VGG16,ResNet
def get_Model(model_value):

    model_value = model_value.lower()
    model =''
    if  model_value == 'squeezenet':
        model = SqueezeNet().get_model()
    elif model_value =='alexnet':
        model = AlexNet().get_model()

    elif model_value == 'vgg16':
        model = VGG16().get_model()

    elif model_value == 'resnet':
        model = ResNet().get_model() 


    return model  



def training(model_value,isDAactive, isPretrainedActive,epochsNumber):


    train_loader,validation_loader,test_loader = load_dataset(True)

    # Ci saranno tre classi presenti
    model = get_Model(model_value)

    model = trainval_classifier(model, isPretrainedActive, model_value, train_loader, validation_loader, lr=0.1, exp_name=model_value, momentum=0.99, epochs=epochsNumber)

    #test (viene passato il modello appena allenato)
    predictions,labels = test_classifier(model,test_loader)
    accuracy = accuracy_score(labels,predictions)*100
    print("Accuracy of : ",model_value," : ",accuracy)
    return accuracy,model

if  __name__ =='__main__':
     training('SqueezeNet',False,True,5)         

