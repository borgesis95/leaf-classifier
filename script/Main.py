# default mean and std needed by pretrained models from pytorch
from script.Testing import test_classifier
from script.Trainer import trainval_classifier
from script.models import SqueezeNet,AlexNet,ResNet,GoogleNet
import os
from sklearn.metrics import accuracy_score
from script.utils.utils import load_dataset
from matplotlib import pyplot as plt

def get_Model(model_value,feature_extraction):

    model_value = model_value.lower()
    model =''
    if  model_value == 'squeezenet':
        model = SqueezeNet().get_model(feature_extraction = feature_extraction)
    elif model_value =='alexnet':
        model = AlexNet().get_model(feature_extraction = feature_extraction)
    elif model_value == 'resnet':
        model = ResNet().get_model(feature_extraction = feature_extraction)
    elif model_value == 'googlenet':
        model = GoogleNet().get_model(feature_extraction = feature_extraction)
    return model  



def training(model_name,load_dict,epochs_number,lr = 0.001,feature_extr = False,data_augmentation=False):


    train_loader,validation_loader,test_loader,train_dataset = load_dataset(data_augmentation=data_augmentation)
    model = get_Model(model_name,feature_extr)

    model = trainval_classifier(model, load_dict, model_name, train_loader, validation_loader, lr=lr, exp_name=model_name, momentum=0.99, epochs=epochs_number,feature_extraction=feature_extr,data_augmentation = data_augmentation)

    
    predictions,labels = test_classifier(model,test_loader)
    accuracy = accuracy_score(labels,predictions)*100
    print("Accuracy of : ",model_name," : ",accuracy)
    return accuracy,model


def norm(im):
    im = im- im.min()
    return im/im.max()




if  __name__ =='__main__':
   training('resnet',True,50,lr=0.002,feature_extr=False,data_augmentation=True)    
#   train_loader,validation_loader,test_loader,train_dataset = load_dataset(data_augmentation=True)

#   plt.figure(figsize=(12,4))
#   for i in range(10):
#      plt.subplot(2,5,i+1)
#      plt.imshow(norm(train_dataset[0][0].numpy().transpose(1,2,0)))
#   plt.show()
