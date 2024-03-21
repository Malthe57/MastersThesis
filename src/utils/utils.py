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

# if __name__ == '__main__':
#     print(os.getcwd())
#     for i in range(1,6):
#         make_dirs(f"models/classification/C_MIMO/M{i}/")