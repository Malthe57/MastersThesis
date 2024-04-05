import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from data.OneD_dataset import train_collate_fn
import os
from utils.utils import make_dirs

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

        if standardise:
            for column in X.columns:
                max = X[column].max()
                min = X[column].min()
                X[column] = 2*(X[column].values - min)/(max-min)-1
            y_min = y.min().values[0]
            y_max = y.max().values[0]
            y = 2*(y.values - y_min)/(y_max-y_min)-1

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


def prepare_crime(standardise=True):
    if os.path.exists("data/multidimdata/crimedata"):
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

        

        if standardise:
            for column in X.columns:
                max = X[column].max()
                min = X[column].min()
                X[column] = 2*(X[column].values - min)/(max-min)-1

                
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

def load_multireg_data(dataset, ):
    if dataset=="newsdata":
        prepare_news()
        df_train = pd.read_csv("data/multidimdata/newsdata/news_train_data.csv")
        df_val = pd.read_csv("data/multidimdata/newsdata/news_val_data.csv")
        df_test = pd.read_csv("data/multidimdata/newsdata/news_test_data.csv")

        train_array = df_train.values
        val_array = df_val.values
        test_array = df_val.values
        x_train, y_train = train_array[:,:-1], train_array[:,-1]
        x_val, y_val = val_array[:,:-1], val_array[:,-1]
        x_test, y_test = test_array[:,:-1], test_array[:,-1]
        input_dim = x_train.shape[1]
        test_length = x_test.shape[0]
        traindata = MultiDataset(x_train, y_train)
        valdata = MultiDataset(x_val, y_val)
        testdata = MultiDataset(x_test, y_test)
        return traindata, valdata, testdata, input_dim, test_length
    
    elif dataset=='crimedata':
        prepare_crime()
        df_train = pd.read_csv("data/multidimdata/crimedata/crime_train_data.csv")
        df_val = pd.read_csv("data/multidimdata/crimedata/crime_val_data.csv")
        df_test = pd.read_csv("data/multidimdata/crimedata/crime_test_data.csv")

        train_array = df_train.values
        val_array = df_val.values
        test_array = df_test.values
        x_train, y_train = train_array[:,:-1], train_array[:,-1]
        x_val, y_val = val_array[:,:-1], test_array[:,-1]
        x_test, y_test = test_array[:,:-1], test_array[:,-1]
        input_dim = x_train.shape[1]
        test_length = x_test.shape[0]
        traindata = MultiDataset(x_train, y_train)
        valdata = MultiDataset(x_val, y_val)   
        testdata = MultiDataset(x_test, y_test)
        return traindata, valdata, testdata, input_dim, test_length

if __name__ == "__main__":
    prepare_news(overwrite=True)

    df_train = pd.read_csv("data/multidimdata/newsdata/news_train_data.csv")

    train_array = df_train.values
    x_train, y_train = train_array[:,:-1], train_array[:,-1]

    traindata = MultiDataset(x_train, y_train)

    trainloader = DataLoader(traindata, batch_size=8*3, shuffle=True, collate_fn=lambda x: train_collate_fn(x, 3), drop_last=True, pin_memory=True)

    x_sample, y_sample = next(iter(trainloader))
    print(x_sample.shape, y_sample.shape)

    



