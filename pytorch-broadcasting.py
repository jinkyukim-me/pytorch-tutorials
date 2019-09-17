import torch
# Same Shape
m1 = torch.FloatTensor([[3, 2]])
m2 = torch.FloatTensor([[2, 3]])
print('Same Shape')
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = ', m1+m2)
print()
# Vector + Scalar
m1 = torch.FloatTensor([[2,3]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print('Vector + Scalar')
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = ', m1+m2)
print()