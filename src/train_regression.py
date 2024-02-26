import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize_mimo import plot_loss
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork, VarMIMONetwork
from utils.utils import seed_worker, set_seed, init_weights
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.make_dataset import make_toydata
import pandas as pd
import os
import hydra

#define training functions
def train_regression(model, optimizer, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10):

    losses = []
    val_losses = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            model.train()

            x_,y_ = x_.float(), y_.float()

            optimizer.zero_grad()

            output, individual_outputs = model(x_)
            loss = nn.functional.mse_loss(individual_outputs, y_)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float(), val_y.float()
                    val_output, val_individual_outputs = model(val_x)
                    val_loss = nn.functional.mse_loss(val_individual_outputs, val_y)
                    val_loss_list.append(val_loss.item())

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")

    return losses, val_losses

def train_var_regression(model, optimizer, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10):

    losses = []
    val_losses = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            model.train()

            x_,y_ = x_.float(), y_.float()

            optimizer.zero_grad()

            mu, sigma, mus, sigmas = model(x_)
            loss = torch.nn.GaussianNLLLoss(reduction='mean')(mus, y_, sigmas)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float(), val_y.float()
                    val_mu, val_sigma, val_mus, val_sigmas = model(val_x)
                    val_loss = torch.nn.GaussianNLLLoss(reduction='mean')(val_mus, val_y, val_sigmas)
                    val_loss_list.append(val_loss.item())

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")

    return losses, val_losses

@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:

#init hyperparameters
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
        traindata = ToyDataset(x_train, y_train)
        
        x_val, y_val = np.array(list(df_val['x'])), np.array(list(df_val['y']))
        valdata = ToyDataset(x_val, y_val)
        

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
                model = None
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
            plot_loss(losses, val_losses)


if __name__ == "__main__":
    main()