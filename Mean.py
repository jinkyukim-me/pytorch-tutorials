import torch
t = torch.FloatTensor([1,2]) # 32-bit floating point CPU tensor
t1 = torch.DoubleTensor([1,2]) # 64-bit floating point CPU tensor
print('t.mean() = ', t.mean())
print('t1.mean() = ', t1.mean())

# Can't use mean() on integers
t = torch.IntTensor([1,2]) # 32-bit Inter(signed) CPU tensor
t1 = torch.LongTensor([1,2]) # 64-bit Inter(signed) CPU tensor
try:
    print('t.mean() = ', t.mean())
except Exception as E:
    print(E)
try:
    print('t1.mean() = ', t1.mean())
except Exception as E:
    print(E)
print()
t = torch.FloatTensor([[1,2],[3,4]])
print('t = \n', t)
print('t.mean() = ', t.mean())
print('t.mean(dim=0) = ', t.mean(dim=0))
print('t.mean(dim=1) = ', t.mean(dim=1))
print('t.mean(dim=-1) = ', t.mean(dim=-1))
print('t.mean(dim=0, keepdim=True) = ', t.mean(dim=0, keepdim=True))
print('t.mean(dim=1, keepdim=True) = \n', t.mean(dim=1, keepdim=True))
print('t.mean(0, True) = ', t.mean(0, True))
print('t.mean(1, True) = \n', t.mean(1, True))

print()
t = torch.FloatTensor([[1,2,3,4],[5,6,7,8],[5,4,3,2],[9,7,5,3]])
print('t = \n', t)
print('torch.mean(input, dim, keepdim=False, out=None) → Tensor')
print('torch.mean(t) = ', torch.mean(t))
print('torch.mean(t,0) = ', torch.mean(t,0))
print('torch.mean(t,1) = ', torch.mean(t,1))
print('torch.mean(t,0,True) = ', torch.mean(t,0,True))
print('torch.mean(t,1,True) = \n', torch.mean(t,1,True))
try:
    print('torch.mean(t,2) = ', torch.mean(t,2))
except Exception as E:
    print(E)