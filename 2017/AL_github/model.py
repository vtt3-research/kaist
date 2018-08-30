import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=256):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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
        self.drop =  nn.Dropout()
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.ReLU6 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.fc7 = nn.Linear(4096, 4096)
        self.ReLU7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.ReLU6(x)
        x = self.drop(x)
        x = self.fc7(x)
        x = self.ReLU7(x)
        x = self.fc8(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
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