import torch
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
print(x)
print(x.shape)
print(y)
print(y.shape)
print()
print(torch.cat([x,y], dim=0))
print(torch.cat([x,y], dim=0).shape)
print(torch.cat([x,y], dim=1))
print(torch.cat([x,y], dim=1).shape)