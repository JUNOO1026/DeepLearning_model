import torch
from torch import nn


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.basic = nn.Sequential(nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                                   nn.BatchNorm2d(out_channels, eps=0.01),
                                   nn.ReLU())

    def forward(self, x):
        x = self.basic(x)

        return x


class InceptionNet(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3re, ch3x3, ch5x5re, ch5x5, poolproj):
        super().__init__()

        self.branch1 = BasicConvBlock(in_channels, ch1x1, kernel_size=1, stride=1, padding=1)

        self.branch2 = nn.Sequential(BasicConvBlock(in_channels, ch3x3re, kernel_size=1, stride=1, padding=1),
                                     BasicConvBlock(ch3x3re, ch3x3, kernel_size=3, stride=1, padding=1))

        self.branch3 = nn.Sequential(BasicConvBlock(in_channels, ch5x5re, kernel_size=1, stirde=1, padding=1),
                                     BasicConvBlock(ch5x5re, ch5x5, kernel_size=5, stride=1, padding=1))

        self.brach4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                    BasicConvBlock(in_channels, poolproj, kernel_size=1, stride=1, padding=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]

        return torch.cat(output, dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, bais=False, drop_p=0.7, **kwargs):
        super().__init__()

        self.avgpool = nn.AvgPool2d(5, stride=3, padding=0)
        self.conv = BasicConvBlock(in_channels, 128, kernel_size=1, padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class InceptionNetV1(nn.Module):
    def __init__(self, num_classes=1000, use_aux=True, init_weights=None, drop_p=0.4, drop_p_aux=0.7):
        super().__init__()

        self.use_aux = use_aux

        self.conv1 = BasicConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 =