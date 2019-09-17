import torch
# Same Shape
m1 = torch.FloatTensor([[3, 2]])
m2 = torch.FloatTensor([[2, 3]])
print('Same Shape')
print('m1 = ', m1)
print('m2 = ', m2)
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = ', m1+m2)
print()
# Vector + Scalar
m1 = torch.FloatTensor([[2,3]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print('Vector + Scalar')
print('m1 = ', m1)
print('m2 = ', m2)
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = ', m1+m2)
print()
# 1 x 2 Vector + 2 x 1 Vector
m1 = torch.FloatTensor([[2,3]]) # [[2,3]] -> [[2,3],[2,3]]
m2 = torch.FloatTensor([[3],[4]]) # [[3],[4]] -> [[3,3],[4,4]]
print('1 x 2 Vector + 2 x 1 Vector')
print('m1 = ', m1)
print('m2 = \n', m2)
print('Size of m1 = ', m1.shape)
print('Size of m2 = ', m2.shape)
print('m1 + m2 = \n', m1+m2)
print()