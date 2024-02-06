import torch
from torch import nn
from torchinfo import summary


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
    def __init__(self, in_channels, ch1x1, ch3x3re, ch3x3, ch5x5re, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConvBlock(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(BasicConvBlock(in_channels, ch3x3re, kernel_size=1),
                                     BasicConvBlock(ch3x3re, ch3x3, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(BasicConvBlock(in_channels, ch5x5re, kernel_size=1),
                                     BasicConvBlock(ch5x5re, ch5x5, kernel_size=5, padding=2))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     BasicConvBlock(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]

        return torch.cat(output, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, bias=False, drop_p=0.7, **kwargs):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConvBlock(in_channels, 128, kernel_size=1)
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
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = BasicConvBlock(64, 64, kernel_size=1)
        self.conv3 = BasicConvBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = InceptionNet(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionNet(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionNet(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionNet(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionNet(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionNet(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionNet(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionNet(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionNet(832, 384, 192, 384, 48, 128, 128)
        # 채널 값 수정 (torchinfo summary channel error 해결)
        if use_aux:
            self.inceptionAux1 = InceptionAux(512, num_classes)
            self.inceptionAux2 = InceptionAux(528, num_classes)
        else:
            self.inceptionAux1 = None
            self.inceptionAux2 = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.inceptionAux1 is not None and self.training:
            inceptionaux1 = self.inceptionAux1(x)
        else:
            inceptionaux1 = None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.inceptionAux2 is not None and self.training:
            inceptionaux2 = self.inceptionAux2(x)
        else:
            inceptionaux2 = None
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, inceptionaux1, inceptionaux2


model = InceptionNetV1()
summary(model, input_size=(2, 3, 224, 224), device="cuda")


