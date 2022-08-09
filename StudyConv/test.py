import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

image_path = "imgs/dog1.png"
image = Image.open(image_path)
image = image.convert("RGB")
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


# 创建网络模型
class Chz(nn.Module):
    def __init__(self):
        super(Chz, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


chz = Chz()
chz.load_state_dict(torch.load("chz9.pth"))
print(chz)
image = torch.reshape(image, (1, 3, 32, 32))
chz.eval()
with torch.no_grad():
    output = chz(image)
print(output)
print(output.argmax(1))
