import torch
import torch.nn as nn
input = torch.Tensor(1,1,28,28) #batch_size=1, channel=1, height=5, width=5
conv1 = nn.Conv2d(1,5,5) #input_channel=1, output_channel=5, kernel_size=5
pool = nn.MaxPool2d(2) #kernel_size=2
out = conv1(input)
out2 = pool(out)
print('out.size() = ', out.size())
print('out2.size() = ', out2.size())
