from torchvision.models import squeezenet1_0
from torch import nn


def get_Squeezenet_model(num_classes):
    model = squeezenet1_0(pretrained=True)
    num_class = num_classes
    model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 3
    return model