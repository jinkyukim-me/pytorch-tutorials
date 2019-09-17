import torch
print('Multiplication vs Matrix Multiplication')
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print('m1 = \n', m1)
print('m2 = \n', m2)
print('Shape of Matrix 1 = ', m1.shape) # 2 x 2
print('Shape of Matrix 2 = ', m2.shape) # 2 x 1
print('m1.matmul(m2) = \n', m1.matmul(m2))
print('m1 * m2 = \n', m1 * m2)
print('m1.mul(m2) = \n', m1.mul(m2))
