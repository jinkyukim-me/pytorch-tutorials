import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    def len(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y

dataset = CustomDataset()

# DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
