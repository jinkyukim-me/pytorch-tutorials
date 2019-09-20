import torch
import torch.nn as nn
'''
input image size : 227 x 227
filter size = 11 x 11
stride = 4
padding = 0
'''
conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
print('conv = ', conv)
inputs = torch.Tensor(1, 1, 227, 227)
print(inputs.shape)