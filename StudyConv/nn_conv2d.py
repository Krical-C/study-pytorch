import torch
from torch import nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./Data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        return x


myModule = MyModule()
print(myModule)

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    img, target = data
    output = myModule(img)
    print(img.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_image("input", img, step, dataformats="NCHW")
    # torch.Size([64, 6, 30, 30])-->[xx,3,30,30]
    output = torch.reshape(output, [-1, 3, 30, 30])
    print(output)
    writer.add_image("output", output, step, dataformats="NCHW")
    step = step + 1
writer.close()
