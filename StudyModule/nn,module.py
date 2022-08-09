import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


myModule = MyModule()
x = torch.tensor(1.0)
output = myModule.forward(x)
print(output)
