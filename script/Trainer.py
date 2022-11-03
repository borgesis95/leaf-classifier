from pickletools import optimize
import torch
import os  
from torch import nn
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time

class AvgMeter():
    """ Calculates the loss and accuracy on individual batches"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, value, num):
        self.sum += value*num
        self.num += num

    def value(self):
        try:
            return self.sum/self.num
        except:
            return None


def trainval_classifier(model,loadcheckpoint,modelName,train_loader,validation_loader,exp_name='experiment', lr=0.001, epochs=50, momentum=0.99,logdir='logs',feature_extraction = False,
data_augmentation = False,external_dataset = False):
    
    print("Learning rate --> ",lr)
    print("Feature extraction --> ",feature_extraction)
    since = time.time()

    # Se il modello è stato salvato (e se il flag loadcheckpoint è presente), allora verrà caricato il modello
    # allenato precedentemente
    if loadcheckpoint:
        chekcpoint_path = './checkpoint/external='+ str(external_dataset)+'_' + modelName +'_lr=' +str(lr)+'_fe='+str(feature_extraction)+'_da='+str(data_augmentation)+'_checkpoint.pth';
        if(os.path.isfile(chekcpoint_path)):
            print("loading checkpoint:")
            model.load_state_dict(torch.load(chekcpoint_path)['state_dict'])   
       

    # -- Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = custom_optimizer(model,lr,momentum=momentum,feature_extraction=feature_extraction)

    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    summary_path = join(logdir,'external='+str(external_dataset)+'_'+exp_name+"_lr="+str(lr)+'_fe='+str(feature_extraction)+'_da='+str(data_augmentation))
    writer = SummaryWriter(summary_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    loader = {
        'train': train_loader,
        'valid' : validation_loader
}

    def save(model,epoch,lr):
        print("saving checkpoint...");

        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        torch.save({
            'state_dict' : model.state_dict(),
            'epoch' : epoch
        }, "{}{}_{}_{}.pth".format('checkpoint\\','external='+ str(external_dataset),exp_name,"lr="+str(lr) +'_fe='+str(feature_extraction)+'_da='+str(data_augmentation),'checkpoint'))

    
    global_step = 0
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        print('-' * 10)

         # Saranno effettuate due iterazioni una per il training ed una per il validation
        for mode in ['train' , 'valid']:
            loss_meter.reset(); 
            acc_meter.reset();
            model.train() if mode == 'train' else model.eval()
          

            # Abilitazione dei gradienti

            with torch.set_grad_enabled(mode=='train'):
                for i, batch in enumerate(loader[mode]):
                    x = batch[0].to(device)
                    y = batch[1].to(device)


                    n = x.shape[0]
                    global_step+=n
                    output = model(x)
                    loss = criterion(output,y)

                    if mode == 'train':
                        loss.backward()
                        torch.cuda.empty_cache()
                        
                        optimizer.step()
                        optimizer.zero_grad()
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    # numero di elementi nel batch
                    n = batch[0].shape[0] 
                    loss_meter.add(loss.item(),n)
                    acc_meter.add(acc,n)

                    if mode == 'train':
                        writer.add_scalar('loss/train', loss_meter.value(),global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(),global_step=global_step)

                writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
                writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
       
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, loss_meter.value(), acc_meter.value()))
        save(model,e,lr)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model



# Need to build an optmizer that is able to just update only 
#desidered parameters 
def custom_optimizer(model,lr,momentum,feature_extraction):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    params_to_update = model.parameters()

    if feature_extraction:
        parmams_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                parmams_to_update.append(param)
                print("Required \t",name)
            else :
                print(" Not required: \t",name)

    c_optimizer = SGD(params_to_update,lr,momentum)
    return c_optimizer
    

