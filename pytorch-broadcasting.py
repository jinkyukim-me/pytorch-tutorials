import torch
# same shape
m1 = torch.FloatTensor([[3, 2]])
m2 = torch.FloatTensor([[2, 3]])
print('same shape')
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = ', m1+m2)
