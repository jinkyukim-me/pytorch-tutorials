import torch
ft = torch.FloatTensor([0, 1, 2])
print('ft = ', ft)
print('ft.shape = ', ft.shape)
print('ft.unsqueeze(0) = ', ft.unsqueeze(0))
print('ft.unsqueeze(0).shape = ', ft.unsqueeze(0).shape)
print('ft.unsqueeze(1) = \n', ft.unsqueeze(1))
print('ft.unsqueeze(1).shape = ', ft.unsqueeze(1).shape)
print('ft.unsqueeze(-2) = ', ft.unsqueeze(-2))
print('ft.unsqueeze(-2).shape = ', ft.unsqueeze(-2).shape)
print('ft.unsqueeze(-1) = \n', ft.unsqueeze(-1))
print('ft.unsqueeze(-1).shape = ', ft.unsqueeze(-1).shape)

