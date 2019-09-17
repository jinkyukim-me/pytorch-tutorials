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