import torch

t = torch.FloatTensor([[1,2],[3,4]])
print('t = \n', t)
print()
print('t.max() = ', t.max())
print()
print('t.max(dim=0) = \n', t.max(dim=0))
print('Max = ', t.max(dim=0)[0])
print('Argmax = ', t.max(dim=0)[1])
print()
print('t.max(dim=1) = \n', t.max(dim=1))
print('Max = ', t.max(dim=1)[0])
print('Argmax = ', t.max(dim=1)[1])