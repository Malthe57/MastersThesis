import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from src.visualization.visualize_mimo import plot_loss
from models.mimo import MIMONetwork, NaiveNetwork, C_MIMONetwork, C_NaiveNetwork
from utils.utils import seed_worker, set_seed, init_weights
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn


# useful functions 🤖

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
                torch.save(model, f'{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")

    return losses, val_losses

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
    torch.save(torch.stack(val_checkpoint_list), f'{model_name}_checkpoints.pt')

    return losses, val_losses, val_checkpoint_list


if __name__ == "__main__":
        #init hyperparameters
        set_seed(1)

        model_name = "MIMO"
        mode = "Regression"
        n_subnetworks = 3
        n_hidden_units = 32
        n_hidden_units2 = 128 
        learning_rate = 3e-4
        M = 3
        naive=False
        #Set generator seed
        g = torch.Generator()
        g.manual_seed(0)
        train_epochs = 500
        val_every_n_epochs = 2
        batch_size = 60

        

        #load data
        if mode == "Regression":
            lower = -0.25
            upper = 1.0
            std = 0.02

            #train
            N_train = 2000
            x_train, y_train = generate_data(N_train, lower, upper, std)
            traindata = ToyDataset(x_train, y_train)
            #val
            N_val = 500
            x_val, y_val = generate_data(N_val, lower, upper, std)
            valdata = ToyDataset(x_val, y_val)
            #test
            N_test = 500
            x_test, y_test = generate_data(N_test, lower, upper, std)
            testdata = ToyDataset(x_test, y_test)

            if naive == False:
                trainloader = DataLoader(traindata, batch_size=batch_size*M, shuffle=True, collate_fn=lambda x: train_collate_fn(x, M), drop_last=True, worker_init_fn=seed_worker, generator=g)
                valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)
                testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)

            else:
                trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=True, worker_init_fn=seed_worker, generator=g)
                valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)
                testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)
            
            #load model
            model = MIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2)
            model.apply(init_weights)
            optimizer = torch.optim.adam(model.parameters(), lr=learning_rate)

            #train model
            losses, val_losses = train_regression(model, optimizer, trainloader, valloader, train_epochs, model_name, val_every_n_epochs)

        elif mode == "Classification":
            traindata, valdata, testdata = load_cifar("../data/")
            if naive == False:
                trainloader = DataLoader(traindata, batch_size=batch_size*M, shuffle=True, collate_fn=lambda x: C_train_collate_fn(x, M), drop_last=True, worker_init_fn=seed_worker, generator=g)
                valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)
                testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)
                model = MIMONetwork(n_subnetworks=M)
            else:
                trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: C_Naive_train_collate_fn(x, M), drop_last=True, worker_init_fn=seed_worker, generator=g)
                valloader = DataLoader(traindata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)
                testloader = DataLoader(traindata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)
                Naive_model = NaiveNetwork(n_subnetworks=M)
                
            model.apply(init_weights)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=3e-4)
            loss_fn = nn.NLLLoss(reduction='mean')

            losses, val_losses, val_checkpoint_list = losses, val_losses, val_checkpoint_list = train_classification(model, optimizer, trainloader, valloader, epochs=30, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=20, loss_fn = loss_fn, device=device)
        

        #plot loss
        plot_loss(losses, val_losses)