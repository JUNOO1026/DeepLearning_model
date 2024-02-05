import torch
from torch import nn
from torchinfo import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion))

        self.projection = projection
        self.act = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        print('residual : ', residual.shape)

        if self.projection is not None:
            shortcut = self.projection(x)
            print(shortcut.shape)
        else:
            shortcut = x

        output = self.act(residual + shortcut)
        print(output.shape)

        return output


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, projection=None, **kwargs):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                                bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                                                stride=stride, bias=False),
                                      nn.BatchNorm2d(out_channels * self.expansion)
                                      )

        self.projection = projection
        self.act = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        output = self.act(residual + shortcut)

        return output


class ResNet(nn.Module):
    def __init__(self, block, block_list, num_classes=1000, zero_init_residual=True, **kwargs):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_layers(block, 64, block_list[0], stride=1)
        self.stage2 = self.make_layers(block, 128, block_list[1], stride=2)
        self.stage3 = self.make_layers(block, 256, block_list[2], stride=2)
        self.stage4 = self.make_layers(block, 512, block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def make_layers(self, block, out_channels, num_block, stride=1, **kwargs):
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))
        else:
            projection = None

        layers = []
        layers += [block(self.in_channels, out_channels, stride, projection)]

        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_block):
            layers += [block(self.in_channels, out_channels)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        print(x.shape)
        x = self.stage2(x)
        print(x.shape)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)


model = resnet18()
summary(model, input_size=(2, 3, 224, 224), device='cuda')
