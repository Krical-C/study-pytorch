import torch
from torch.nn import L1Loss

input = torch.tensor([1,2,3],dtype=torch.float32)
targets=torch.tensor([1,2,5],dtype=torch.float32)
input=torch.reshape(input,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=L1Loss()
result=loss(input,targets)
print(result)