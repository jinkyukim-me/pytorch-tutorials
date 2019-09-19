# imports
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# For reproducibility
torch.manual_seed(1)
# Traing Data
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # |x_data|=(6,2) m=6 d=2
y_data = [[0], [0], [0], [1], [1], [1]] # |y_data|=(6,1)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)
# Computing the Hypothesis
print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))