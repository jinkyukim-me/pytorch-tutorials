import torch
import numpy as np
z = torch.FloatTensor([1,2,3])
m = np.argmax(z)
n = np.argmin(z)
print(m)
print(n)

