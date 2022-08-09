import torch
import torchvision

# 方式一加载
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式二加载
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# # model = torch.load("vgg16_method2.pth")
# print(vgg16)


# 陷阱一
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = torch.load("myModule_method1.pth")
print(model)
