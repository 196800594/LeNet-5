import torch
from torch.nn import Module
from torch import nn

class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
