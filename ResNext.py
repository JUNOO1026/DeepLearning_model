import torch
from torch import nn
from torchinfo import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels * self.expansion, padding=1, kernel_size=3, bias=False),
                                      nn.BatchNorm2d(out_channels * self.expansion))

        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        print(residual.shape)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        output = self.relu(residual + shortcut)

        return output


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, cardinality, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=cardinality, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels * self.expansion))

        self.projection = projection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        print("residual :", residual.shape)

        if self.projection is not None:
            shortcut = self.projection(x)
            print("shortcut :",  shortcut.shape)
        else:
            shortcut = x

        output = self.relu(residual + shortcut)

        return output


class ResNext(nn.Module):
    def __init__(self, block, block_list, cardinality, num_classes=1000, zero_init=True):
        super().__init__()

        self.in_channels = 64
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_layers(block, 64, block_list[0], stride=1)
        self.stage2 = self.make_layers(block, 128, block_list[1], stride=2)
        self.stage3 = self.make_layers(block, 256, block_list[2], stride=2)
        self.stage4 = self.make_layers(block, 512, block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_init:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        print("x : ", x.shape)
        x = self.stage2(x)
        print("x : ", x.shape)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def make_layers(self, block, out_channels, block_list, stride=1):
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1),
                nn.BatchNorm2d(out_channels * block.expansion))
        else:
            projection = None

        layers = []
        layers += [block(self.in_channels, out_channels, self.cardinality, stride, projection)]

        self.in_channels = out_channels * block.expansion

        for _ in range(1, block_list):
            layers += [block(self.in_channels, out_channels, self.cardinality)]

        return nn.Sequential(*layers)


def ResNext50(**kwargs):
    return ResNext(BottleNeck, [3, 4, 6, 3], cardinality=32)


model = ResNext50()

summary(model, input_size=(2, 3, 224, 224), device='cuda')

# 일반적으로 nn.ReLU()에서 nn.ReLU(inplace=True)를 사용해서 메모리 효율성을 높여주는 것이긴한데, 이런 느낌은? 좋지 않음.
# 쉽게 설명하면 어떤 메모리에 input data를 삭제하고 output을 집어넣어서 메모리의 효율성을 높여주는 방식임. 그러나 그리 좋은 건 아닌듯
# backpropagation을 통해 weight를 업데이트하는데 문제가 충분히 발생할 수 있다고 함.