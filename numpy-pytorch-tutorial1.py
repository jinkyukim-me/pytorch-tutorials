import numpy as np
import torch
# 0D Array with Numpy
t = np.array(5)
print('0D Array with Numpy')
print('t = ', t)
print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print()
# 1D Array with NumPy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print('1D Array with NumPy')
print('t = ', t)
print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing
print()

# 2D Array with NumPy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print('2D Array with NumPy')
print('t = \n', t)
print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print('t[:, 1] = ', t[:, 1])
print('Shape of t[:, 1] = ' ,t[:, 1].shape)
print('t[:, :-1] = \n', t[:, :-1])
print()

# 3D Array with NumPy
t = np.array([[[1,2,3,4], [2,4,5,6]], [[3,4,5,6],[4,5,6,7]],
             [[5,6,7,8],[6,7,8,9]]])
print('3D Array with NumPy')
print('t = \n', t)
print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print()

# 1D Array with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print('1D Array with PyTorch')
print('t = ', t)
print('Rank of t: ', t.dim()) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print('Size of t: ', t.size()) # 크기 출력
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing
print()

# 2D Array with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])
print('2D Array with PyTorch')
print('t = ', t)
print('Rank of t: ', t.dim()) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print('Size of t: ', t.size()) # 크기 출력
print('t[:, 1] = ', t[:, 1])
print('Size of t[:, 1] = ', t[:, 1].size())
print('t[:, :-1] = \n', t[:, :-1]) # :-1에서 슬라이싱할 때 -1은 포함되지 않음에 주의
print()

# 3D Array with PyTorch
t = torch.FloatTensor([[[1,2,3,4], [2,4,5,6]],
                       [[3,4,5,6],[4,5,6,7]],
                        [[5,6,7,8],[6,7,8,9]]])
print('3D Array with PyTorch')
print('t = \n', t)
print('Rank of t: ', t.dim()) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력
print()