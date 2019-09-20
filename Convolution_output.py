import torch
import torch.nn as nn

# 1
'''
input image size : 227 x 227
filter size = 11 x 11
stride = 4
padding = 0
'''
# filter size
conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
print('conv = ', conv)
# input image size
inputs = torch.Tensor(1, 1, 227, 227)
print('inputs.shape = ', inputs.shape)
# output image size
out = conv(inputs)
print('out=conv(inputs) = \n', out)
print('out.shape = ', out.shape)
print()
# 2
'''
input image size : 64 x 64
filter size = 7 x 7
stride = 2
padding = 0
'''
# filter size
conv = nn.Conv2d(1, 1, 7, stride=2, padding=0)
print('conv = ', conv)
# input image size
inputs = torch.Tensor(1, 1, 64, 64)
print('inputs.shape = ', inputs.shape)
# output image size
out = conv(inputs)
print('out=conv(inputs) = \n', out)
print('out.shape = ', out.shape)
print()
# 3
'''
input image size : 32 x 32
filter size = 5 x 5
stride = 1
padding = 2
'''
# filter size
conv = nn.Conv2d(1, 1, 5, stride=1, padding=2)
print('conv = ', conv)
# input image size
inputs = torch.Tensor(1, 1, 32, 32)
print('inputs.shape = ', inputs.shape)
# output image size
out = conv(inputs)
print('out=conv(inputs) = \n', out)
print('out.shape = ', out.shape)
print()
# 4
'''
input image size : 32 x 64
filter size = 5 x 5
stride = 1
padding = 0
'''
# filter size
conv = nn.Conv2d(1, 1, 5, stride=1, padding=0)
print('conv = ', conv)
# input image size
inputs = torch.Tensor(1, 1, 32, 64)
print('inputs.shape = ', inputs.shape)
# output image size
out = conv(inputs)
print('out=conv(inputs) = \n', out)
print('out.shape = ', out.shape)
print()
# 5
'''
input image size : 64 x 32
filter size = 3 x 3
stride = 1
padding = 1
'''
# filter size
conv = nn.Conv2d(1, 1, 3, stride=1, padding=1)
print('conv = ', conv)
# input image size
inputs = torch.Tensor(1, 1, 64, 32)
print('inputs.shape = ', inputs.shape)
# output image size
out = conv(inputs)
print('out=conv(inputs) = \n', out)
print('out.shape = ', out.shape)
print()