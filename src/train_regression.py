import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize import plot_loss, plot_log_probs
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork, VarMIMONetwork, VarNaiveNetwork
from models.bnn import BayesianNeuralNetwork, BayesianConvNeuralNetwork
from utils.utils import seed_worker, set_seed, init_weights
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn, bnn_collate_fn
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from train_loops import train_regression, train_var_regression, train_BNN
from data.make_dataset import make_toydata
import pandas as pd
import os
import hydra


def main_mimo(cfg: dict) -> None:
    config = cfg.experiments["hyperparameters"]
    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name = config.model_name
    naive=config.is_naive
    is_var = config.is_var
    plot = config.plot

    
    #model parameters
    n_subnetworks = config.n_subnetworks
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size
    
    if naive:
        print(f"Training Naive model with {n_subnetworks} subnetworks on regression task.")
        model_name = "Naive/" + config.model_name + f'_{config.n_subnetworks}_members'
    else:
        if n_subnetworks == 1:
            print(f"Training baseline model on regression task.")
            model_name = "MIMO/" + config.model_name
        else:
            print(f"Training MIMO model with {n_subnetworks} subnetworks on regression task.")
            model_name = "MIMO/" + config.model_name + f'_{config.n_subnetworks}_members'


    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs    

    make_toydata()
        
    #train
    df_train = pd.read_csv('data/toydata/train_data.csv')
    df_val = pd.read_csv('data/toydata/val_data.csv')
    
    x_train, y_train = np.array(list(df_train['x'])), np.array(list(df_train['y']))
    traindata = ToyDataset(x_train, y_train, normalise=True)
    
    x_val, y_val = np.array(list(df_val['x'])), np.array(list(df_val['y']))
    valdata = ToyDataset(x_val, y_val, normalise=True)

    if naive == False:
        trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: test_collate_fn(x, n_subnetworks), drop_last=False)

        #load model
        if is_var: 
            model = VarMIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
        else:
            model = MIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
        

    else:
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: naive_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, n_subnetworks), drop_last=False)

        if is_var:
            model = VarNaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
        else:
            model = NaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
    
    #load model
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #train model
    if is_var:
        losses, val_losses = train_var_regression(model, optimizer, trainloader, valloader, train_epochs, model_name, val_every_n_epochs)
    else:
        losses, val_losses = train_regression(model, optimizer, trainloader, valloader, train_epochs, model_name, val_every_n_epochs)
    if plot == True:
        plot_loss(losses, val_losses, model_name=model_name, task='regression')

def main_bnn(cfg: dict) -> None:
    config = cfg.experiments["hyperparameters"]
    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name =  "BNN/" + config.model_name 
    plot = config.plot

    #model parameters
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size
    print(f"Training BNN model on regression task.")
    
    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs    

    make_toydata()
        
    #train
    df_train = pd.read_csv('data/toydata/train_data.csv')
    df_val = pd.read_csv('data/toydata/val_data.csv')
    
    x_train, y_train = np.array(list(df_train['x'])), np.array(list(df_train['y']))
    traindata = ToyDataset(x_train, y_train, normalise=True)
    
    x_val, y_val = np.array(list(df_val['x'])), np.array(list(df_val['y']))
    valdata = ToyDataset(x_val, y_val, normalise=True)

    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=bnn_collate_fn, drop_last=True, pin_memory=True)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, collate_fn=bnn_collate_fn, drop_last=True, pin_memory=True)

    model = BayesianNeuralNetwork(n_hidden_units, n_hidden_units2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN(model, optimizer, trainloader, valloader, train_epochs, model_name, val_every_n_epochs)
    if plot:
        plot_loss(losses, val_losses, model_name=model_name, task='regression')
        plot_log_probs(log_priors, log_variational_posteriors, NLLs)

@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:

#init hyperparameters
    config = cfg.experiments["hyperparameters"]

    mode = config.mode

    match mode:
        case 0: # baseline
            cfg.experiments["hyperparameters"].n_subnetworks = 1
            main_mimo(cfg)
        case 1: # MIMO
            main_mimo(cfg)
        case 2: # Naive multiheaded
            cfg.experiments["hyperparameters"].is_naive = True
            main_mimo(cfg)
        case 3: # BNN
            main_bnn(cfg)
        case 4: # Combined MIMO and BNN
            raise NotImplementedError
        case 9: # Old MIMO (with one output)
            main_mimo(cfg)



if __name__ == "__main__":
    main()

    # mode = 3
    # match mode:
    #     case 0: # baseline
    #         raise NotImplementedError
    #     case 1: # MIMO
    #         print(1)
    #     case 2: # Naive multiheaded
    #         print(2)
    #     case 3: # BNN
    #         raise NotImplementedError
    #     case 4: # Combined MIMO and BNN
    #         raise NotImplementedError
    #     case 9: # Old MIMO (with one output)
    #         print(9)

    # main()