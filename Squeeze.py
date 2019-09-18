import torch
ft = torch.FloatTensor([[0], [1], [2]])
print('ft = \n', ft)
print('ft.dim() = ', ft.dim())
print('ft.shape = ', ft.shape)
print()
print('ft.squeeze() = ', ft.squeeze())
print('ft.squeeze().shape = ', ft.squeeze().shape)
print()
print('ft.squeeze(dim=0) = \n', ft.squeeze(dim=0)) # 3이기때문에 변화가 없다
print('ft.squeeze(dim=0).shape = ', ft.squeeze(dim=0).shape)
print()
print('ft.squeeze(dim=1) = ', ft.squeeze(dim=1)) # 1이기때문에 squeeze되었다
print('ft.squeeze(dim=1).shape = ', ft.squeeze(dim=1).shape)