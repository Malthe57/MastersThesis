import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize_mimo import plot_loss, plot_log_
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork, VarMIMONetwork
from models.bnn import BayesianConvNeuralNetwork
from utils.utils import seed_worker, set_seed, init_weights
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.make_dataset import make_toydata
from train_loops import train_classification, train_BNN_classification
import pandas as pd
import os
import hydra


def main_mimo(cfg):
    config = cfg.experiments["hyperparameters"]

    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name = config.model_name
    naive=config.is_naive
    plot = config.plot

    #model parameters
    n_subnetworks = config.n_subnetworks
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size
    

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar("data/")
    if naive == False:
        trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: C_train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, n_subnetworks), drop_last=False)
        model = C_MIMONetwork(n_subnetworks=n_subnetworks)
    else:
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: C_Naive_train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, n_subnetworks), drop_last=False)
        model = C_NaiveNetwork(n_subnetworks=n_subnetworks)
        
    model.apply(init_weights)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=3e-4)
    loss_fn = nn.NLLLoss(reduction='mean')

    losses, val_losses, val_checkpoint_list = train_classification(model, optimizer, trainloader, valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=20, loss_fn = loss_fn, device=device)
    if plot==True:
        plot_loss(losses, val_losses)

    def main_bnn(cfg):
        config = cfg.experiments["hyperparameters"]

    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name = config.model_name
    naive=config.is_naive
    plot = config.plot

    #model parameters
    n_subnetworks = config.n_subnetworks
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size
    

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar("data/")
    CIFAR_trainloader = DataLoader(traindata, batch_size=500, shuffle=True, pin_memory=True)
    CIFAR_valloader = DataLoader(valdata, batch_size=500, shuffle=True, pin_memory=True)

    BNN_model = BayesianConvNeuralNetwork(hidden_units1=32, hidden_units2=128, channels1=32, channels2=64, device=device)
    BNN_model = BNN_model.to(device)
    optimizer = torch.optim.Adam(BNN_model.parameters(), lr=1e-4)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN_classification(BNN_model, optimizer, CIFAR_trainloader, CIFAR_valloader, epochs=30, model_name='C_BNN', val_every_n_epochs=5, device=device)

    if plot == True:
        plot_loss(losses, val_losses)

    

    
    @hydra.main(config_path="../conf/", config_name="config.yaml", version_base=1.2)
    def main(cfg: dict) -> None:
        config = cfg.experiments["hyperparameters"]
        mode = config.mode

        match mode:
            case 0: #baseline
                cfg.experiments["hyperparameters"].n_subnetworks = 1
                main_mimo(cfg)
            case 1: #MIMO
                main_mimo(cfg)
            case 2: #Naive multi-headed
                cfg.experiments["hyperparameters"].is_naive = True 
                main_mimo(cfg)
            case 3: #BNN
                main_bnn(cfg)
            case 4: #combined BNN and MIMO
                NotImplementedError

    if __name__ == "__main__":
        main()