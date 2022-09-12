from pickletools import optimize
import torch
import os  
from torch import nn
from torch.optim import SGD
from torchnet.meter import AverageValueMeter
# from torchnet.logger import VisdomPlotLogger, VisdomSaver
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from os.path import join

def trainval_classifier(model,pretrained,modelName,train_loader,validation_loader,exp_name='experiment', lr=0.001, epochs=50, momentum=0.99,logdir='logs'):
    if pretrained:
        if(os.path.isfile('checkpoint\\' + modelName + 'checkpoint.pth')):
            model.load_state_dict(torch.load('checkpoint\\' + modelName + '_checkpoint.pth')['state_dict'])   

    # -- Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr,momentum=momentum)

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    writer = SummaryWriter(join(logdir,exp_name))

    #Plot su Visdom
    # loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend':['train','valid']})
    # acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy','legend':['train','valid']})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device --> ",device)
    model.to(device)
    
    loader = {
        'train': train_loader,
        'valid' : validation_loader
    }

    def save_checkpoint(model, epoch):
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        torch.save({
            'state_dict' : model.state_dict(),
            'epoch' : epoch
        }, "{}{}_{}.pth".format('checkpoint\\',exp_name, 'checkpoint'))

    
    # global_step = 0
    for e in range(epochs):
         # Saranno effettuate due iterazioni una per il training ed una per il validation
        for mode in ['train' , 'valid']:
            loss_meter.reset(); acc_meter.reset();
            model.train() if mode == 'train' else model.eval()

            # Abilitazione dei gradienti

            with torch.set_grad_enabled(mode=='train'):
                for i, batch in enumerate(loader[mode]):
                    x = batch[0].to(device)
                    y = batch[1].to(device)

                    n = x.shape[0]
                    # global_step+=n
                    output = model(x)
                    l = criterion(output,y)

                    if mode == 'train':
                        # Calcolo dei gradienti
                        l.backward()
                        # Ottimiziamo i gradienti
                        optimizer.step()
                        optimizer.zero_grad()
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    n = batch[0].shape[0]
                    loss_meter.add(l.item()*n,n)
                    acc_meter.add(acc*n,n)

                    # if mode == 'train':
                        # writer.add_scalar('loss/train', loss_meter.value(),global_step=global_step)
                        # writer.add_scalar('accuracy/train', acc_meter.value(),global_step=global_step)

        print("Epoca numero: ",e)
        save_checkpoint(model, e )

    return model