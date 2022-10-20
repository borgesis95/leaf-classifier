# default mean and std needed by pretrained models from pytorch
from script.Testing import test_classifier
from script.Trainer import trainval_classifier
from script.models import SqueezeNet,AlexNet,VGG16,ResNet,GoogleNet

from sklearn.metrics import accuracy_score
from script.utils.utils import load_dataset

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
    elif model_value == 'googlenet':
        model = GoogleNet().get_model()
    return model  



def training(model_value,loadCheckpoint,epochsNumber,lr = 0.001):


    train_loader,validation_loader,test_loader = load_dataset()

    # Ci saranno tre classi presenti
    model = get_Model(model_value)

    model = trainval_classifier(model, loadCheckpoint, model_value, train_loader, validation_loader, lr=lr, exp_name=model_value, momentum=0.99, epochs=epochsNumber)

    #test (viene passato il modello appena allenato)
    predictions,labels = test_classifier(model,test_loader)
    accuracy = accuracy_score(labels,predictions)*100
    print("Accuracy of : ",model_value," : ",accuracy)
    return accuracy,model

if  __name__ =='__main__':
   
    training('googlenet',True,25)         

