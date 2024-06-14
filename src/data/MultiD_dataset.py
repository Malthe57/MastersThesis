import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from data.OneD_dataset import train_collate_fn
import os
from utils.utils import make_dirs
import matplotlib.pyplot as plt

def generate_multidim_data(N, lower, upper, std, dim=1, num_points_to_remove=0, projection_matrix=None, save_x_path=None):

    # create data
    x_1d = np.linspace(lower, upper, N)
    
    # noise std from ]-inf, 0.5] and noise 5*std from [0.5, inf[
    n1 = len(x_1d[x_1d>=0.5])
    noise_range = np.linspace(0, 1, n1)
    noise1 = np.random.normal(0, 1, N-n1) * std
    noise2 = np.random.normal(0,1, n1) *(1 + noise_range * 4) * std
    noise = np.concatenate((noise1, noise2))

    # Regression data function
    y = x_1d + 0.3 * np.sin(2*np.pi * (x_1d + noise)) + 0.3 * np.sin(4 * np.pi * (x_1d + noise)) + noise
    # y = x_1d + 0.3 * np.sin(2*np.pi * (x_1d)) + 0.3 * np.sin(4 * np.pi * (x_1d)) + noise

    # project to multidimensional space
    if dim > 1:
        x = np.dot(x_1d[:,None], projection_matrix)
    else: 
        x = x_1d
    
    if num_points_to_remove > 0:
        start1 = (len(x) // 3) - (num_points_to_remove//2)
        # end = start + num_points_to_remove
        end1 = start1 + num_points_to_remove//2 - 50
        start2 = end1 + 50
        end2 = start2 + num_points_to_remove//2 + 50
        delete_indices = np.array(list(range(start1,end1)) + list(range(start2,end2)))
        x_1d = np.delete(x_1d, delete_indices, axis=0)
        x  = np.delete(x, delete_indices, axis=0)
        y = np.delete(y, delete_indices, axis=0)

    if save_x_path is not None:
        np.savez(f"{save_x_path}/x_1d.npz", x_1d=x_1d)

    return x, y

def prepare_news(standardise = True, overwrite = False):
    # fetch dataset 
    if os.path.exists("data/multidimdata/newsdata") and overwrite==False:
        print("Data already exists")
        pass
    else:
        make_dirs("data/multidimdata/newsdata")
        online_news_popularity = fetch_ucirepo(id=332) 
        
        # data (as pandas dataframes) 
        X = online_news_popularity.data.features 
        y = online_news_popularity.data.targets 

        

        X['shares'] = y

        #set rng-generator seed
        rng = np.random.default_rng(seed=0)
        X_vals = X.values
        #shuffle rows
        rng.shuffle(X_vals, axis=0)
        #divide into train, val and test
        data_length = len(X_vals)
        train_data = X_vals[:int(data_length*0.7), :]
        val_data = X_vals[int(data_length*0.7):int(data_length*0.8), :]
        test_data = X_vals[int(data_length*0.8):,:]

        #save dataset to csv for quick loading
        pd.DataFrame(train_data).to_csv("data/multidimdata/newsdata/news_train_data.csv", index=False)
        pd.DataFrame(val_data).to_csv("data/multidimdata/newsdata/news_val_data.csv", index=False)
        pd.DataFrame(test_data).to_csv("data/multidimdata/newsdata/news_test_data.csv", index=False)


def prepare_crime(standardise=True, overwrite=False):
    if os.path.exists("data/multidimdata/crimedata") and overwrite==False:
        print("Data already exists")
        pass
    else:
        make_dirs("data/multidimdata/crimedata")
        # fetch dataset 
        communities_and_crime = fetch_ucirepo(id=183) 
        
        # data (as pandas dataframes) 
        data = communities_and_crime.data.features 
        y = communities_and_crime.data.targets 
        # Remove features containing missing values, except 'state'
        data = data.drop(data.loc[:, data.columns != 'state'].columns[data.loc[:, data.columns != 'state'].eq('?').any()], axis=1)
        X = data.drop(['state', 'communityname','fold'], axis=1)

        

        # if standardise:
        #     for column in X.columns:
        #         max = X[column].max()
        #         min = X[column].min()
        #         X[column] = 2*(X[column].values - min)/(max-min)-1

                
        X['targets'] = y.values
    
        #set rng-generator seed
        rng = np.random.default_rng(seed=0)
        X_vals = X.values
        #shuffle rows
        rng.shuffle(X_vals, axis=0)
        #divide into train, val and test
        data_length = len(X_vals)
        train_data = X_vals[:int(data_length*0.7), :]
        val_data = X_vals[int(data_length*0.7):int(data_length*0.8), :]
        test_data = X_vals[int(data_length*0.8):,:]

        #save dataset to csv for quick loading
        pd.DataFrame(train_data).to_csv("data/multidimdata/crimedata/crime_train_data.csv", index=False)
        pd.DataFrame(val_data).to_csv("data/multidimdata/crimedata/crime_val_data.csv", index=False)
        pd.DataFrame(test_data).to_csv("data/multidimdata/crimedata/crime_test_data.csv", index=False)



class MultiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x = self.x[idx,:]
        y = self.y[idx]
        return x, y
    def __len__(self):
        return len(self.x)

def load_multireg_data(dataset, num_points_to_remove=0, standardise=True, ood=False):
    if dataset=="multitoydata":

        path = 'data/multidimdata/toydata'
        if num_points_to_remove > 0:
            path += f"{num_points_to_remove}_points_removed"

        df_train = pd.read_csv(f'{path}/train_data.csv')
        df_val = pd.read_csv(f'{path}/val_data.csv')
        df_test = pd.read_csv(f'{path}/test_data.csv')

    elif dataset=="newsdata":
        prepare_news()
        df_train = pd.read_csv("data/multidimdata/newsdata/news_train_data.csv")
        df_val = pd.read_csv("data/multidimdata/newsdata/news_val_data.csv")
        df_test = pd.read_csv("data/multidimdata/newsdata/news_test_data.csv")

    elif dataset=='crimedata':
        prepare_crime()
        df_train = pd.read_csv("data/multidimdata/crimedata/crime_train_data.csv")
        df_val = pd.read_csv("data/multidimdata/crimedata/crime_val_data.csv")
        df_test = pd.read_csv("data/multidimdata/crimedata/crime_test_data.csv")

    min = 0
    max = 0
        
    for i, column in enumerate(df_train.columns):
        max = df_train[column].max()
        min = df_train[column].min()
        if standardise:
            df_train[column] = 2*(df_train[column].values - min)/(max-min)-1
            df_val[column] = 2*(df_val[column].values - min)/(max-min)-1
            df_test[column] = 2*(df_test[column].values - min)/(max-min)-1

    train_array = df_train.values
    val_array = df_val.values
    test_array = df_test.values
    x_train, y_train = train_array[:,:-1], train_array[:,-1]
    x_val, y_val = val_array[:,:-1], val_array[:,-1]
    x_test, y_test = test_array[:,:-1], test_array[:,-1]
    input_dim = x_train.shape[1]
    test_length = x_test.shape[0]

    if ood and dataset=="crimedata":
        np.random.seed(1871)
        # Add noise to the features of the test data
        x_noise = np.random.normal(0, 0.5, x_test.shape)
        x_test += x_noise

    traindata = MultiDataset(x_train, y_train)
    valdata = MultiDataset(x_val, y_val)
    testdata = MultiDataset(x_test, y_test)
    return traindata, valdata, testdata, input_dim, test_length, max, min

if __name__ == "__main__":

    load_multireg_data("crimedata", num_points_to_remove=0, standardise=True)


    # prepare_news(overwrite=True)

    # df_train = pd.read_csv("data/multidimdata/newsdata/news_train_data.csv")

    # train_array = df_train.values
    # x_train, y_train = train_array[:,:-1], train_array[:,-1]

    # traindata = MultiDataset(x_train, y_train)

    # trainloader = DataLoader(traindata, batch_size=8*3, shuffle=True, collate_fn=lambda x: train_collate_fn(x, 3), drop_last=True, pin_memory=True)

    # x_sample, y_sample = next(iter(trainloader))
    # print(x_sample.shape, y_sample.shape)


    



