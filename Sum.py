import torch

t = torch.FloatTensor([[1,2],[3,4]])
print('t = \n',t)
print('Sum')
print('t.sum() = ', t.sum())
print('t.sum(dim=0) = ', t.sum(dim=0))
print('t.sum(dim=1) = ', t.sum(dim=1))
print('t.sum(dim=-2) = ', t.sum(dim=-2))
print('t.sum(dim=-1) = ', t.sum(dim=-1))
print('t.sum(dim=0, keepdim=True) = ', t.sum(dim=0, keepdim=True))
print('t.sum(dim=1, keepdim=True) = \n', t.sum(dim=1, keepdim=True))
print()
print('Sum')
print('torch.sum(t)', torch.sum(t))
print('torch.sum(t, 0) = ', torch.sum(t, 0))
print('torch.sum(t, 1) = ', torch.sum(t, 1))
print('torch.sum(t, -2) = ', torch.sum(t, -2))
print('torch.sum(t, -1) = ', torch.sum(t, -1))
print('torch.sum(t, 0, True) = ', torch.sum(t, 0, True))
print('torch.sum(t, 1, True) = \n', torch.sum(t, 1, True))


