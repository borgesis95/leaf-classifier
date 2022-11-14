# default mean and std needed by pretrained models from pytorch
from script.Testing import test_classifier
from script.Trainer import trainval_classifier
from script.models import SqueezeNet,AlexNet,ResNet,GoogleNet
import os
from sklearn.metrics import accuracy_score
from script.utils.utils import load_dataset,write_reacaps
from matplotlib import pyplot as plt

def get_Model(model_value,feature_extraction):
    model_value = model_value.lower()
    model =''

    print("modelvalue",model_value)
    if  model_value == 'squeezenet':
        model = SqueezeNet().get_model(feature_extraction = feature_extraction)
    elif model_value =='alexnet':
        model = AlexNet().get_model(feature_extraction = feature_extraction)
    elif model_value == 'resnet':
        model = ResNet().get_model(feature_extraction = feature_extraction)
    elif model_value == 'googlenet':
        model = GoogleNet().get_model(feature_extraction = feature_extraction)
   
    return model  





def train(model_name,load_checkpoint,external,epochs,lr = 0.001,feature_extr = False,data_augmentation=False):
    train_loader,validation_loader,test_loader,train_dataset = load_dataset(data_augmentation=data_augmentation,training_csv='train.new',validation_csv='validation.new',test_csv='test.new')
    model = get_Model(model_name,feature_extr)
   
    momentum = 0.99
    print("Learning rate: ",lr)
    print("Feature extraction:",feature_extr)

    PATH = model_name+'_ep='+str(epochs)+'_feature_extraction=_'+str(feature_extr)+'_da='+str(data_augmentation)+'_external='+str(external)+'_lr=' +str(lr)
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

    MODEL_NAME="resnet"
    LOAD_CHECKPOINT = True
    EXTERNAL = False,
    EPOCHS = 25
    LEARNING_RATE = 0.001
    FEATURE_EXTR = True
    DATA_AUGMENTATION = True

    train(MODEL_NAME,LOAD_CHECKPOINT,EXTERNAL,EPOCHS,lr=LEARNING_RATE,feature_extr=FEATURE_EXTR,data_augmentation=DATA_AUGMENTATION)    
    show_image()

 
