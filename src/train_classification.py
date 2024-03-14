import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize import plot_loss, plot_log_probs
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork, VarMIMONetwork, MIMOWideResnet
from models.bnn import BayesianConvNeuralNetwork
from models.mimbo import MIMBOConvNeuralNetwork
from utils.utils import seed_worker, set_seed, init_weights
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.make_dataset import make_toydata
from training_loops import train_classification, train_BNN_classification
import pandas as pd
import os
import hydra
import wandb


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
    is_resnet = config.is_resnet
    weight_decay = config.weight_decay

    if naive:
        if is_resnet:
            depth = config.depth
            widen_factor = config.widen_factor
            p = config.dropout_rate
            print(f"Training Naive WideResnet({depth}, {widen_factor}) model with {n_subnetworks} subnetworks on classification task.")
            model_name = 'C_NaiveWide' + f'_{depth}_{widen_factor}_{n_subnetworks}_members'
        else:
            print(f"Training Naive model with {n_subnetworks} subnetworks on classification task.")
            model_name = "C_Naive/" + config.model_name + f'_{n_subnetworks}_members'

    else:
        if is_resnet:
            depth = config.depth
            widen_factor = config.widen_factor
            p = config.dropout_rate
            print(f"Training MIMO WideResnet({depth}, {widen_factor}) model with {n_subnetworks} subnetworks on classification task.")
            model_name = f"C_MIMOWide_{depth}_{widen_factor}/" + 'C_MIMOWide' + f'_{depth}_{widen_factor}_{n_subnetworks}_members'
        elif n_subnetworks == 1:
            print(f"Training baseline model on classification task.")
            model_name = "C_MIMO/" + config.model_name
        else:
            print(f"Training MIMO model with {n_subnetworks} subnetworks on classification task.")
            model_name = "C_MIMO/" + config.model_name + f'_{n_subnetworks}_members'
    
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
        model = MIMOWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p) if is_resnet else C_MIMONetwork(n_subnetworks=n_subnetworks)
    else:
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: C_Naive_train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, n_subnetworks), drop_last=False)
        model = C_NaiveNetwork(n_subnetworks=n_subnetworks)
        
    model.apply(init_weights)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss(reduction='mean')

    losses, val_losses, val_checkpoint_list = train_classification(model, optimizer, trainloader, valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=20, loss_fn = loss_fn, device=device)
    if plot==True:
        plot_loss(losses, val_losses, model_name=model_name, task='classification')

def main_bnn(cfg):
    config = cfg.experiments["hyperparameters"]

    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name = "C_BNN/" + config.model_name
    plot = config.plot

    #model parameters
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size

    print(f"Training BNN model  on classification task.")

    
    pi = config.pi
    sigma1 = torch.exp(torch.tensor(config.sigma1))
    sigma2 = torch.exp(torch.tensor(config.sigma2))

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar("data/")
    CIFAR_trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    CIFAR_valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    

    BNN_model = BayesianConvNeuralNetwork(hidden_units1=128, channels1=32, channels2=64, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    BNN_model = BNN_model.to(device)
    optimizer = torch.optim.Adam(BNN_model.parameters(), lr=learning_rate)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN_classification(BNN_model, optimizer, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, device=device)

    if plot == True:
        plot_loss(losses, val_losses, model_name=model_name, task='classification')
        plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name=model_name, task='classification')

def main_mimbo(cfg : dict) -> None:
    config = cfg.experiments["hyperparameters"]

    seed = config.seed
    set_seed(seed)

    #Select model to train
    model_name = "C_MIMBO/" + config.model_name + f'_{config.n_subnetworks}_members'
    plot = config.plot

    #model parameters
    learning_rate = config.learning_rate
    
    batch_size = config.batch_size
    n_subnetworks = config.n_subnetworks
    print(f"Training MIMBO model with {n_subnetworks} subnetworks on classification task.")

    pi = config.pi
    sigma1 = torch.exp(torch.tensor(config.sigma1))
    sigma2 = torch.exp(torch.tensor(config.sigma2))

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar("data/")
    CIFAR_trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, collate_fn=lambda x: C_train_collate_fn(x, n_subnetworks), shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    CIFAR_valloader = DataLoader(valdata, batch_size=batch_size, collate_fn=lambda x: C_test_collate_fn(x, n_subnetworks), shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    

    MIMBO_model = MIMBOConvNeuralNetwork(n_subnetworks=n_subnetworks, hidden_units1=128, channels1=32, channels2=64, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    MIMBO_model = MIMBO_model.to(device)
    optimizer = torch.optim.Adam(MIMBO_model.parameters(), lr=learning_rate)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN_classification(MIMBO_model, optimizer, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, device=device)

    if plot == True:
        plot_loss(losses, val_losses, model_name=model_name, task='classification')
        plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name=model_name, task='classification')


@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:
    config = cfg.experiments["hyperparameters"]
    mode = config.mode
    
    mode = config.mode
    is_resnet = config.is_resnet
    if config.model_name == 'C_BNN':
        name = f"{config.model_name}_classification"
    else:
        if is_resnet:
            depth = config.depth
            widen_factor = config.widen_factor
            name = f"{config.model_name}_{depth}_{widen_factor}_{config.n_subnetworks}_members_classification"
        else:
            name = f"{config.model_name}_{config.n_subnetworks}_members_classification"
    
    wandb.init(
        project="MastersThesis", 
        name=name, 
           
        config={
            "model_name": config.model_name,
            "mode": config.mode,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "train_epochs": config.train_epochs,

        })
    
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
        case 4: # MIMBO
            main_mimbo(cfg)

if __name__ == "__main__":
    main()