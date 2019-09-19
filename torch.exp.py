import torch
print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))
print('e^2 equals: ', torch.exp(torch.FloatTensor([2])))
print('e^1, e^2, e^3 equals: ', torch.exp(torch.FloatTensor([1,2,3])))
