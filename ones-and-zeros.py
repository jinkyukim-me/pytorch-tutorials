import torch
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
print(x.shape)
print()
print(torch.ones_like(x))
print(torch.ones_like(x).shape)
print()
print(torch.zeros_like(x))
print(torch.zeros_like(x).shape)
print()
print(torch.ones(3,2,4))
print(torch.ones(3,2,4).shape)
print()
print(torch.zeros(3,2,4))
print(torch.zeros(3,2,4).shape)