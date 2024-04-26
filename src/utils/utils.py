import numpy as np
import random
import torch
from torch import nn
import pandas as pd
import numpy as np
import os

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
def get_training_min_max():
    df_train = pd.read_csv('data/toydata/train_data.csv')
    x_train, y_train = np.array(list(df_train['x'])), np.array(list(df_train['y']))
    
    x_train_min = np.min(x_train)
    x_train_max = np.max(x_train)
    y_train_min = np.min(y_train)
    y_train_max = np.max(y_train)

    return x_train_min, x_train_max, y_train_min, y_train_max

def make_dirs(directory):
    # directories 

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_zero_mean_mixture_variance(sigma1, sigma2, pi):
    """
    assuming both means are 0.
    """
    return pi * sigma1**2 + (1 - pi) * sigma2**2

def compute_weight_decay(sigma):
    return 1 / (2 * sigma**2)

def logmeanexp(x, dim=None, keepdim=False):
    """
    Stable computation of log(mean(exp(x)) 
    Inputs:
        x: input tensor (log probabilities)
        dim: dimension to reduce
        keepdim: keep the reduced dimension or not
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        to_numpy = True

    if dim is None:
        x, dim = x.view(-1), 0
    
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))

    x = x if keepdim else x.squeeze(dim)
    if to_numpy:
        x = x.numpy()   
        
    return x

if __name__ == '__main__':

    sigmas = [0.1, 0.5, 1, 3, 5, 7.5, 10, 32.62, 50]
    for i in sigmas:
        print(compute_weight_decay(i))