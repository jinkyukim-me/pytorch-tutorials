import torch
x = torch.FloatTensor([[1,2], [3,4]])
print(x)
print(x.mul(2.))
print(x.mul(4))
print(x)
print()
print(x.mul_(10))
print(x)