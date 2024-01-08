from torch import nn
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 12 * 12, 10)

    def forward(self, x):
        
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

myawesomemodel = MyAwesomeModel()