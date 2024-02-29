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


def train_classification(model, optimizer, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10, checkpoint_every_n_epochs=20, loss_fn = nn.NLLLoss(reduction='mean'), device='cpu'):
    losses = []
    val_losses = []
    val_checkpoint_list = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            x_,y_ = x_.float().to(device), y_.long().to(device)

            model.train()

            optimizer.zero_grad()

            log_prob, _, _ = model(x_)
            
            # sum loss per subnetwork
            # mean is already taken over the batch, because we use reduction = 'mean' in the loss function
            loss = 0
            for log_p, y in zip(log_prob, y_.T):
                # print(log_p.shape)
                # print(y.shape)
                loss += loss_fn(log_p, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float().to(device), val_y.long().to(device)
                    log_prob, _, _ = model(val_x)
                
                    val_loss = 0
                    for log_p, y in zip(log_prob, val_y.T):
                        val_loss += loss_fn(log_p, y)

                    val_loss_list.append(val_loss.item())
                if (e+1) % checkpoint_every_n_epochs == 0:
                    val_checkpoint_list.append(log_prob[0,:,:])

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
    torch.save(torch.stack(val_checkpoint_list), f'models/C_{model_name}_checkpoints.pt')

    return losses, val_losses, val_checkpoint_list


@hydra.main()
def main(cfg: dict) -> None:
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

    losses, val_losses, val_checkpoint_list = losses, val_losses, val_checkpoint_list = train_classification(model, optimizer, trainloader, valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=20, loss_fn = loss_fn, device=device)
    if plot==True:
        plot_loss(losses, val_losses)