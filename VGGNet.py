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

        # 위의 방법 즉, VGGNet에서 kaiming_normal_ 방식으로 weight를 initiallize한 이유는
        # kaiming_normal 방식의 경우 ReLU activation과 같이 사용하며 그래디언트 완화 목적임.

        # 여기서 trunc 가중치 초기화 방법을 굳이 안쓰는 이유는 다양한 크기의 커널을 병렬로 사용하는 InceptionNet과 다르며
        # 네트워크의 성능에 대한 안정성을 유지하기 위해 각 가중치를 특정 구간으로 제한하는 trunc_normal 방식은 별로임.

        # 예를들어, 동일한 커널로 학습되는 VGGNet의 경우 특정 커널 사이즈로 인해 네트워크 성능이 떨어지진 않음.

        # 하지만, InceptionNet 처럼 여러 커널 사이즈로 학습이 되는 경우에는, 특정 커널 사이즈의 영향도가 너무 커버리면
        # 다른 커널 사이즈로 학습된 부분이 무시되는 경향이 있을 수 있으므로 어느정도 특정 구간을 제한하여 값을 샘플링하는 것임.

        # if init_weights:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #             nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)

        # 예를들어 분산이 1인 random variable이 들어간다고 하면 각 weight가 곱해지고 더해지면 출력 노드의 variance(변화)는 커지게 됨. 이러면 학습에 도움을 주지 않음.
        # 이러한 문제를 막기 위해 Back Propagation에서의 분산을 1/N_out 으로 나눠주고 그 nonlinearity는 relu로 학습할 수 있도록 한다.



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



