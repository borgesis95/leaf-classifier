# default mean and std needed by pretrained models from pytorch
from src.Testing import test_classifier
from src.Trainer import trainval_classifier
from src.Models import SqueezeNet,AlexNet,ResNet,GoogleNet
import os
from sklearn.metrics import accuracy_score
from src.utils.utils import load_dataset,write_reacaps
from matplotlib import pyplot as plt
from src.config import MODEL_NAME,LOAD_CHECKPOINT,LEARNING_RATE,FEATURE_EXTR,DATA_AUGMENTATION,EPOCHS

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





def train(model_name,load_checkpoint,epochs,lr = 0.001,feature_extr = False,data_augmentation=False):
    train_loader,validation_loader,test_loader,train_dataset = load_dataset(data_augmentation=data_augmentation)
    model = get_Model(model_name,feature_extr)
   
    momentum = 0.99
    print("Learning rate:",lr)
    print("Feature extraction:",feature_extr)

    PATH = model_name+'_ep='+str(epochs)+'_feature_extraction=_'+str(feature_extr)+'_da='+str(data_augmentation)+'_lr=' +str(lr)
    model,time_elapsed = trainval_classifier(model,load_checkpoint,train_loader,validation_loader, lr,epochs,momentum,feature_extr,PATH)

    predictions,labels = test_classifier(model,test_loader)
    accuracy = accuracy_score(labels,predictions)*100
    print("Accuracy of : ",model_name," : ",accuracy)

    result = model_name + "---accuracy:" + str(accuracy) +'---time----'+ str(time_elapsed)+', '+ PATH
    write_reacaps(result)


def norm(im):
    im = im- im.min()
    return im/im.max()



def show_image():
    train_loader,validation_loader,test_loader,train_dataset = load_dataset(data_augmentation=True,training_csv='train.new',validation_csv='validation.new',test_csv='test.new')

    plt.figure(figsize=(12,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(norm(train_dataset[0][0].numpy().transpose(1,2,0)))
    plt.show()




if  __name__ =='__main__':
    train(MODEL_NAME,LOAD_CHECKPOINT,EPOCHS,lr=LEARNING_RATE,feature_extr=FEATURE_EXTR,data_augmentation=DATA_AUGMENTATION)    
    show_image()

 
