# 모델 설계 코테 준비

import torch
from torch import nn
from torchinfo import summary

cfgs = {
    "A" : [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B" : [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C" : [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D" : [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E" : [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


class VGGNet(nn.Module):
    def __init__(self, cfgs, batch_norm, n_classes=1000, init_weights=True, drop_p=0.5):
        super().__init__()

        self.feature = self.make_layer(cfgs, batch_norm)
        self.adpavgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=drop_p),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=drop_p),
                                        nn.Linear(4096, n_classes))

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:  #  bias가 있다면
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.adpavgpool(x)
        x = torch.flatten(x, start_dim=1) # 그래야 각 이미지에 대한 정보만 하나의 리스트 shape으로 만들 수 있음.
        x = self.classifier(x)

        return x

    def make_layer(self, cfgs, batch_norm):
        layers = []
        in_channels = 3

        for out_channels in cfgs:
            if type(out_channels) == int:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU()]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                               nn.ReLU()]

                in_channels = out_channels

            else:
                layers += [nn.MaxPool2d(2)]

        return nn.Sequential(*layers) #unzip


model = VGGNet(cfgs["D"], batch_norm=False)
summary(model, input_size=(2, 3, 224, 224), device='cuda')
