import torch
from torch import nn
from torchinfo import summary


class BottleNeck(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()

        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels, 4 * k, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(4 * k),
                                      nn.ReLU(),
                                      nn.Conv2d(4 * k, k, kernel_size=3, stride=1, padding=1))
        # Desnet은 layer을 지나면서 해당 feature size를 줄이 않음. 따라서 kernel_size가 3이면 padding 1을 주어야 함.

    def forward(self, x):
        output = torch.cat([x, self.residual(x)], 1)

        return output
    # Desnet은 resnet처럼 skip connection을 할때 output feature에 x를 더하는 것은 정확한 정보를 알 수 없다!
    # 이게 무슨 말이냐면 입력한 x와 convolution을 통과한 feature가 더해져서 들어가므로 x에 대한 정보를 제대로 알 수 없다는 것.
    # 따라서 concat을 통해 해당 feature의 depth를 늘리고 x에 대한 feature를 계속 보게 함으로써 x에 대한 정보를 계속 갖고 갈 수 있다는 장점. resnet에 비해 더 정확한 정보를 전달할 수 있다.


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                        nn.AvgPool2d(2))

        # transition layer에서는 채널 수, 크기도 반으로 줄임.

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, block, k, divide=0.5, num_classes=1000):
        super().__init__()

        self.k = k  # 32
        out_channels = 2 * self.k

        # 첫번째 layer만 conv > batch > relu

        self.conv1 = nn.Sequential(nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers = []

        for num in block[:-1]:
            layers += self.make_layers(out_channels, num)
            out_channels += num * self.k
            ## transition
            out_channels2 = int(out_channels * divide)
            layers += [Transition(out_channels, out_channels2)]
            out_channels = out_channels2

        layers += [self.make_layers(out_channels, block[-1])]
        out_channels += block[-1] * self.k

        layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def make_layers(self, in_channels, n_block):
        bottle_layer = []

        for _ in range(n_block):
            bottle_layer += [BottleNeck(in_channels, self.k)]
            in_channels += self.k
        return nn.Sequential(*bottle_layer)  # unzip


def densnet121(**kwargs):
    return DenseNet([6, 12, 24, 16], k=32, **kwargs)

def densnet169(**kwargs):
    return DenseNet([6, 12, 32, 32], k=32, **kwargs)

def densenet201(**kwargs):
    return DenseNet([6, 12, 48, 32], k=32, **kwargs)

def densenet264(**kwargs):
    return DenseNet([6, 12, 64, 48], k=32, **kwargs)

model = densenet264()


from torchinfo import summary
summary(model, input_size=(2,3,224,224), device='cuda')