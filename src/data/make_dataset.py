# -*- coding: utf-8 -*-
import pandas as pd
import sys
import os
sys.path.append(os.getcwd() + '/src/')
from data.OneD_dataset import generate_data
from data.MultiD_dataset import generate_multidim_data
from utils.utils import set_seed
import numpy as np


def make_toydata():
    if os.path.exists("data/toydata"):
        print('Toy dataset already exists')
        pass
    
    else:
        os.mkdirs("data/toydata")
        
        set_seed(1871)

        df_train = pd.DataFrame(columns=["x", "y"])
        df_val = pd.DataFrame(columns=["x", "y"])
        df_test = pd.DataFrame(columns=["x", "y", "line"])

        # Generate train data
        N_train = 2000
        x, y = generate_data(N_train, lower=-0.25, upper=1, std=0.02)

        # Generate validation data
        N_val = 500
        x_val, y_val = generate_data(N_val, lower=-0.25, upper=1, std=0.02)

        # Generate test data
        N_test = 500
        x_test, y_test = generate_data(N_test, lower=-0.5, upper=1.5, std=0.02)

        _, line = generate_data(N_test, lower=-0.5, upper=1.5, std=0.0)

        df_train['x'] = x
        df_train['y'] = y

        df_val['x'] = x_val
        df_val['y'] = y_val

        df_test['x'] = x_test
        df_test['y'] = y_test
        df_test['line'] = line

        df_train.to_csv("data/toydata/train_data.csv", index=False)
        df_val.to_csv("data/toydata/val_data.csv", index=False)
        df_test.to_csv("data/toydata/test_data.csv", index=False)
        print('Created Toydata successfully')

def make_multidim_toydata():
    if os.path.exists("data/multidimdata/toydata"):
        print('Multidimensional toy dataset already exists')
        pass
    
    else:
        os.mkdirs("data/multidimdata/toydata")
        set_seed(1871)

        dim = 64
        projection_matrix = np.random.randn(1, dim)
        # Generate train data
        N_train = 20000
        x, y = generate_multidim_data(N_train, lower=-0.25, upper=1, std=0.02, dim=dim, projection_matrix=projection_matrix)

        # Generate validation data
        N_val = 5000
        x_val, y_val = generate_multidim_data(N_val, lower=-0.25, upper=1, std=0.02, dim=dim, projection_matrix=projection_matrix)

        # Generate test data
        N_test = 5000
        x_test, y_test = generate_multidim_data(N_test, lower=-0.5, upper=1.5, std=0.02, dim=dim, projection_matrix=projection_matrix)

        df_train = pd.DataFrame(np.concatenate((x,y[:,None]), axis=1))
        df_val = pd.DataFrame(np.concatenate((x_val,y_val[:,None]), axis=1))
        df_test = pd.DataFrame(np.concatenate((x_test,y_test[:,None]), axis=1))
    
        df_train.to_csv("data/multidimdata/toydata/train_data.csv", index=False)
        df_val.to_csv("data/multidimdata/toydata/val_data.csv", index=False)
        df_test.to_csv("data/multidimdata/toydata/test_data.csv", index=False)
        print('Created Multidimensional Toydata successfully')

if __name__ == '__main__':
    make_multidim_toydata()