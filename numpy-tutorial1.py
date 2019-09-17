import numpy as np
import torch

# 1D Array with NumPy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print('1D Array with NumPy')
print(t)

print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing
print()

# 2D Array with NumPy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print('2D Array with NumPy')
print(t)

print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print()

# 1D Array with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print('1D Array with Pytorch')
print(t)

print('Rank of t: ', t.dim()) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print('Size of t: ', t.size()) # 크기 출력

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing