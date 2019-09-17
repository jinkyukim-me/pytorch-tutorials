import numpy as np
import torch

# 1D Array with NumPy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Rank of t: ', t.ndim) # 차원 출력
print('Shape of t: ', t.shape) # 크기 출력

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing
