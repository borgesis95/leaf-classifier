from torchvision.models import squeezenet1_1,AlexNet as AlNet,googlenet,resnet18
from torch import nn

from src.utils.utils import set_parameter_requires_grad

       
class SqueezeNet():
    def get_model(self,num_class = 3, pretrained = True,feature_extraction = False):
        model = squeezenet1_1(pretrained=True)
        set_parameter_requires_grad(model, feature_extraction)
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_class
        print("Run: Squeezenet: ",model)

        return model

class AlexNet():
    def get_model(self,num_class = 3, pretrained = True,feature_extraction = False):
        model = AlNet()
        set_parameter_requires_grad(model, feature_extraction)
        model.classifier[6] = nn.Linear(4096, num_class)
        print("Run: AlexNet: ",model)

        return model

class ResNet():
    def get_model(self,num_class = 3, pretrained = True,feature_extraction = False):
        model = resnet18(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extraction)
        model.fc = nn.Linear(512, num_class)
        model.num_classes = num_class
        print("Run ResNet: ",model)

        return model


class GoogleNet():
    def get_model(self,num_class = 3, pretrained = True,feature_extraction = False):
        model = googlenet(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extraction)
        model.fc = nn.Linear(1024, num_class)
        print("Run GoogleNet: ",model)

        return model

        
