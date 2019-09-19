import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.5
batch_size = 10


