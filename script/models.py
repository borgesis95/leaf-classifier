from torchvision.models import squeezenet1_1,AlexNet as AlNet,vgg16,resnet18
from torch import nn

class SqueezeNet():
    def get_model(self,num_class = 4, pretrained = True):
        model = squeezenet1_1(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_class
        print("Run: Squeezenet: ",model)

        return model

class AlexNet():
    def get_model(self,num_class = 4, pretrained = True):
        model = AlNet()
        model.classifier[6] = nn.Linear(4096, num_class)
        print("Run: AlexNet: ",model)

        return model

class VGG16():
    def get_model(self,num_class = 4, pretrained = True):
        model = vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_class)
        print("Run: VGG16: ",model)

        return model


class ResNet():
    def get_model(self,num_class = 4, pretrained = True):
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_class)
        model.num_classes = num_class
        print("Run: ResNet: ",model)

        return model
