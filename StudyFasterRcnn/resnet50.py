from torch import nn
from torch.nn import Sequential


class Bottleneck(nn.Module):

    def __init__(self, in_channels, filters, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.stride = stride
        F1, F2, F3 = filters
        self.out_channels = F3

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, F1, 1, stride, 0, False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),

            nn.Conv2d(F1, F2, stride, 1, 1, False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),

            nn.Conv2d(F2, F3, 1, 1, 0, False),
            nn.BatchNorm2d(F3)
        )

        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, F3, 1, stride, False),
            nn.BatchNorm2d(F3),
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        # 判断有无下采样操作
        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, [64, 64, 256], 1),
            Bottleneck(256, [64, 64, 256]),
            Bottleneck(256, [64, 64, 256])
        )
        self.stage3 = nn.Sequential(
            Bottleneck(256, [128, 128, 512], 2),
            Bottleneck(521, [128, 128, 128]),
            Bottleneck(521, [128, 128, 128]),
            Bottleneck(521, [128, 128, 128]),
        )
