import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import get_training_min_max


def generate_data(N, lower, upper, std):
    # Regression data function
    f = lambda x, epsilon: x + 0.3 * np.sin(2*np.pi * (x+epsilon)) + 0.3 * np.sin(4 * np.pi * (x+epsilon)) + epsilon

    # create data
    x = np.linspace(lower, upper, N)

    y = []
    for i in range(N):
        epsilon = np.random.normal(0, std)
        y.append(f(x[i], epsilon))
    return x, y

class ToyDataset(Dataset):
    """Custom toy dataset"""

    def __init__(self, x, y, normalise = False):
        self.x = x
        self.y = y
        self.normalise = normalise
        x_min, x_max, y_min, y_max = get_training_min_max()

        if self.normalise:
            normalised_x = 2*(self.x - x_min)/(x_max-x_min)-1
            # normalised_y = 2*(self.y- y_min)/(y_max-y_min)-1
            self.x = normalised_x
            # self.y = normalised_y
            

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
def train_collate_fn(batch, M):
    """Collate function for training MIMO"""
    
    x, y = zip(*batch)
    
    x_chunks = torch.stack(torch.chunk(torch.tensor(np.array(x)), M, dim=0), dim=1)
    x_chunks = x_chunks.reshape(x_chunks.shape[0],-1)
    y_chunks = torch.stack(torch.chunk(torch.tensor(y), M, dim=0), dim=1)

    return x_chunks, y_chunks

def test_collate_fn(batch, M):
    """Collate function for testing MIMO"""
    
    x, y = zip(*batch)
    x = torch.tensor(np.array(x))
    if x.dim() == 1:
        x = x[:,None].repeat(1,M)
    elif x.dim()==2:
        x = x.repeat(1,M)
    y = torch.tensor(y)[:,None].repeat(1,M)

    return x, y

def naive_collate_fn(batch, M):
    """Collate function for naive multiheaded model"""

    x, y = zip(*batch)
    x = torch.tensor(np.array(x))
    if x.dim() == 1:
        x = x[:,None]
    y = torch.tensor(y)[:,None].repeat(1,M)

    return x, y

def bnn_collate_fn(batch):

    x, y = zip(*batch)
    x = torch.tensor(np.array(x))
    if x.dim()==1:
        x = x[:,None]
    
    return x, torch.tensor(y)