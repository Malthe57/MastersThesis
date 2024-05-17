import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize import plot_loss, plot_log_probs
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork, VarMIMONetwork, VarNaiveNetwork
from models.bnn import BayesianNeuralNetwork, BayesianConvNeuralNetwork
from models.mimbo import MIMBONeuralNetwork
from utils.utils import seed_worker, set_seed, init_weights, make_dirs, compute_weight_decay
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn, bnn_collate_fn, load_toydata
from data.MultiD_dataset import MultiDataset, prepare_news, prepare_crime, load_multireg_data
from training_loops import train_regression, train_var_regression, train_BNN
from data.make_dataset import make_toydata, make_multidim_toydata
import pandas as pd
import os
import omegaconf
import hydra
import wandb


def main_mimo(cfg: dict, rep : int, seed : int) -> None:
    config = cfg.experiments["hyperparameters"]
    set_seed(seed)

    #Select model to train
    model_name = config.model_name
    naive=config.is_naive
    is_var = config.is_var
    plot = config.plot
    dataset = config.dataset

    #model parameters
    n_subnetworks = config.n_subnetworks
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    # make relevant dirs
    make_dirs(f"models/regression/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"models/regression/checkpoints/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"reports/figures/losses/regression/{model_name}/{dataset}/M{n_subnetworks}/")

    if naive:
        print(f"Training Naive model with {n_subnetworks} subnetworks on regression task.")
        model_name = "Naive/" + f"{dataset}/M{n_subnetworks}/" + config.model_name + f'_{config.n_subnetworks}_members_rep{rep}'
    else:
        if n_subnetworks == 1:
            print(f"Training baseline model on regression task.")
            model_name = "MIMO/" + f"{dataset}/M{n_subnetworks}/" + config.model_name + f"_rep{rep}"
        else:
            print(f"Training MIMO model with {n_subnetworks} subnetworks on regression task.")
            model_name = "MIMO/" + f"{dataset}/M{n_subnetworks}/" + config.model_name + f'_{config.n_subnetworks}_members_rep{rep}'


    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs
    weight_decay = compute_weight_decay(config.sigma1)  

    if dataset=="1D":
        make_toydata()
        traindata, valdata, _, input_dim, _ = load_toydata(normalise=True)
        kwargs = {'max': 1, 'min': -1}

    elif dataset=="multitoydata":
        make_multidim_toydata()
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}

    elif dataset=="newsdata":
        prepare_news(overwrite=False)
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}
    
    elif dataset=='crimedata':
        prepare_crime(overwrite=False)
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}
 
    if naive == False:
        trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: test_collate_fn(x, n_subnetworks), drop_last=False)

        #load model
        if is_var: 
            model = VarMIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)
        else:
            model = MIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
        

    else:
        trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: naive_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, n_subnetworks), drop_last=False)

        if is_var:
            model = VarNaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)

        else:
            model = NaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
    
    #load model
    # model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    #train model
    if is_var:
        losses, val_losses = train_var_regression(model, optimizer, scheduler, trainloader, valloader, train_epochs, model_name, val_every_n_epochs=config.val_every_n_epochs, **kwargs)
    else:
        losses, val_losses = train_regression(model, optimizer, scheduler, trainloader, valloader, train_epochs, model_name, val_every_n_epochs)
    if plot == True:
        plot_loss(losses, val_losses, model_name=model_name, task='regression')

def main_bnn(cfg: dict, rep : int, seed: int) -> None:
    config = cfg.experiments["hyperparameters"]
    set_seed(seed)

    dataset = config.dataset

    #model parameters
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    learning_rate = config.learning_rate
    sigma = config.sigma1

  # make relevant dirs
    make_dirs(f"models/regression/{config.model_name}/{dataset}/")
    make_dirs(f"models/regression/checkpoints/{config.model_name}/{dataset}/")
    make_dirs(f"reports/figures/losses/regression/{config.model_name}/{dataset}/")

    #Select model to train
    model_name =  f"BNN/{dataset}/" + config.model_name + f"_rep{rep}"
    plot = config.plot

    batch_size = config.batch_size
    print(f"Training BNN model on regression task.")
    
    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs 
    pi = config.pi
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)   

    if dataset=="1D":
        make_toydata()
        traindata, valdata, _, input_dim, _ = load_toydata(normalise=True)
        kwargs = {'max': 1, 'min': -1}

    elif dataset=="multitoydata":
        make_multidim_toydata()
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset, standardise=False)
        kwargs = {'max': max, 'min': min}

    elif dataset=="newsdata":
        prepare_news()
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}
    
    elif dataset=='crimedata':
        prepare_crime()
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}



    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=bnn_collate_fn, drop_last=True, pin_memory=True)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=bnn_collate_fn, drop_last=False, pin_memory=True)

    model = BayesianNeuralNetwork(n_hidden_units, n_hidden_units2, input_dim=input_dim, pi=pi, sigma1=sigma1, sigma2=sigma2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN(model, optimizer, scheduler, trainloader, valloader, train_epochs, model_name, val_every_n_epochs, **kwargs)
    if plot:
        plot_loss(losses, val_losses, model_name=model_name, task='regression')
        plot_log_probs(log_priors, log_variational_posteriors, NLLs)

def main_mimbo(cfg: dict, rep: int, seed: int) -> None:
    config = cfg.experiments["hyperparameters"]
    set_seed(seed)

    dataset = config.dataset

    #model parameters
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    learning_rate = config.learning_rate
    n_subnetworks = config.n_subnetworks
    sigma = config.sigma1

    #Select model to train
    model_name =  "MIMBO/" + f"{dataset}/M{n_subnetworks}/" + config.model_name + f'_{config.n_subnetworks}_members_rep{rep}'
    plot = config.plot
    

    # make relevant dirs
    make_dirs(f"models/regression/{config.model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"models/regression/checkpoints/{config.model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"reports/figures/losses/regression/{config.model_name}/{dataset}/M{n_subnetworks}/")

   
    
    batch_size = config.batch_size
    print(f"Training MIMBO model with {n_subnetworks} subnetworks on regression task.")

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs
    pi = config.pi
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2) 

        
    if dataset=="1D":
        make_toydata()
        traindata, valdata, _, input_dim, _ = load_toydata(normalise=True)
        kwargs = {'max': 1, 'min': -1}

    elif dataset=="multitoydata":
        make_multidim_toydata()
        traindata, valdata, _, input_dim, _, max, min = load_multireg_data(dataset, num_points_to_remove=3000, standardise=False)
        kwargs = {'max': max, 'min': min}

    elif dataset=="newsdata":
        prepare_news()
        traindata, valdata, _, input_dim, _, max , min = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}
    
    elif dataset=='crimedata':
        prepare_crime()
        traindata, valdata, _, input_dim, _, max, min  = load_multireg_data(dataset)
        kwargs = {'max': max, 'min': min}

    trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: train_collate_fn(x, n_subnetworks), drop_last=True, pin_memory=True)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: test_collate_fn(x, n_subnetworks), drop_last=False, pin_memory=True)


    model = MIMBONeuralNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim, pi=pi, sigma1=sigma1, sigma2=sigma2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN(model, optimizer, scheduler, trainloader, valloader, train_epochs, model_name, val_every_n_epochs, **kwargs)
    if plot:
        plot_loss(losses, val_losses, model_name=model_name, task='regression')
        plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name=model_name, task='regression')


@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:
    config = cfg.experiments["hyperparameters"]
    reps = config.repetitions
    for r in range(1,reps+1):
        print(f"Running experiment {r} of {reps}")

        seed = config.seed + r - 1

        mode = config.mode
        if config.model_name == 'BNN':
            name = f"{config.model_name}_regression_rep{r}"
        else:
            name = f"{config.model_name}_{config.n_subnetworks}_members_regression_rep{r}"
        
        wandb.init(
            project="FinalRuns", 
            name=name,
            mode='disabled',
            config=omegaconf.OmegaConf.to_container(cfg),
            group=config.dataset)
        
        match mode:
            case 0: # baseline
                cfg.experiments["hyperparameters"].n_subnetworks = 1
                main_mimo(cfg, r, seed)
            case 1: # MIMO
                main_mimo(cfg, r, seed)
            case 2: # Naive multiheaded
                cfg.experiments["hyperparameters"].is_naive = True
                main_mimo(cfg, r, seed)
            case 3: # BNN
                main_bnn(cfg, r, seed)
            case 4: # MIMBO
                main_mimbo(cfg, r, seed)
            case 9: # Old MIMO (with one output)
                main_mimo(cfg, r)

        wandb.finish()


if __name__ == "__main__":
    main()
    