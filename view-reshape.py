import torch
import numpy as np
t = np.array([[[0,1,2],[3,4,5]], [[6,7,8],[9,10,11]]])
print('t = \n', t)
print('t.shape = ', t.shape)
ft = torch.FloatTensor(t)
print('ft = \n', ft)
print('ft.shape = ', ft.shape)
print()
print('View')
print('ft.view([-1, 3]) = \n', ft.view([-1, 3]))
print('ft.view([-1, 3]).shape = ', ft.view([-1, 3]).shape)
print('ft.view([-1, 4]) = \n', ft.view([-1, 4]))
print('ft.view([-1, 4]).shape = ', ft.view([-1, 4]).shape)
print('ft.view([-1, 6]) = \n', ft.view([-1, 6]))
print('ft.view([-1, 6]).shape = ', ft.view([-1, 6]).shape)
try:
    print('ft.view([-1, 5]) = \n', ft.view([-1, 5]))
except Exception as E:
    print(E)
print('ft.view([-1, 1, 3]) = \n', ft.view([-1, 1, 3]))
print('ft.view([-1, 1, 3]).shape = ', ft.view([-1, 1, 3]).shape)
