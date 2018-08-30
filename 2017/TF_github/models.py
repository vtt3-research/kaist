import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from DA_module import DomainAlignment
import numpy as np
__all__ = ['AlexNet', 'alexnet_da']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, batch_size, split,num_classes=31):
        super(AlexNet, self).__init__()
        self.batch_size = batch_size
        self.split = split
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.DA6 = DomainAlignment(self.batch_size, self.split)
        self.ReLU6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(4096, 4096)
        self.DA7 = DomainAlignment(self.batch_size, self.split)
        self.ReLU7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(4096, num_classes)
        self.DA8 = DomainAlignment(self.batch_size, self.split)

    def forward(self, x, alpha1, alpha2, alpha3):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc6(x)
        x = self.DA6(x, alpha1)
        x = self.ReLU6(x)
        x = self.fc7(x)
        x = self.DA7(x, alpha2)
        x = self.ReLU7(x)
        x = self.fc8(x)
        x = self.DA8(x, alpha3)
        return x

def alexnet_da(pretrained=False, model_root=None, batch_size = 256, split = 1, **kwargs):
    model = AlexNet(batch_size, split, **kwargs)
    own_state = model.state_dict()
    if pretrained:
        # for pytorch alexnet parameters
        state_dict = model_zoo.load_url(model_urls['alexnet'], model_root).items()

        own_state = model.state_dict()

        name, param = state_dict[0]
        param = param.data
        own_state['features.0.weight'].copy_(param)

        name, param = state_dict[1]
        param = param.data
        own_state['features.0.bias'].copy_(param)

        name, param = state_dict[2]
        param = param.data
        own_state['features.3.weight'].copy_(param)

        name, param = state_dict[3]
        param = param.data
        own_state['features.3.bias'].copy_(param)

        name, param = state_dict[4]
        param = param.data
        own_state['features.6.weight'].copy_(param)

        name, param = state_dict[5]
        param = param.data
        own_state['features.6.bias'].copy_(param)

        name, param = state_dict[6]
        param = param.data
        own_state['features.8.weight'].copy_(param)

        name, param = state_dict[7]
        param = param.data
        own_state['features.8.bias'].copy_(param)

        name, param = state_dict[8]
        param = param.data
        own_state['features.10.weight'].copy_(param)

        name, param = state_dict[9]
        param = param.data
        own_state['features.10.bias'].copy_(param)

        name, param = state_dict[10]
        param = param.data
        own_state['fc6.weight'].copy_(param)

        name, param = state_dict[11]
        param = param.data
        own_state['fc6.bias'].copy_(param)

        name, param = state_dict[12]
        param = param.data
        own_state['fc7.weight'].copy_(param)

        name, param = state_dict[13]
        param = param.data
        own_state['fc7.bias'].copy_(param)

    return model
