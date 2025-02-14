import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization.visualize import plot_loss, plot_log_probs
from models.mimo import C_MIMONetwork, C_NaiveNetwork, MIMOWideResnet, NaiveWideResnet
from models.mimo2 import MIMOWideResnet, NaiveWideResnet
from models.bnn import BayesianConvNeuralNetwork, BayesianWideResnet
from models.bnn2 import BayesianWideResnet
from models.mimbo import MIMBOConvNeuralNetwork, MIMBOWideResnet
from models.mimbo2 import MIMBOWideResnet
from utils.utils import seed_worker, set_seed, init_weights, make_dirs, compute_weight_decay, model_summary
from data.OneD_dataset import generate_data, ToyDataset, train_collate_fn, test_collate_fn, naive_collate_fn
from data.CIFAR10 import load_cifar10, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.CIFAR100 import load_cifar100
from data.make_dataset import make_toydata
from training_loops import train_classification, train_BNN_classification, run_MIMO_resnet, run_BNN_resnet
import omegaconf
import pandas as pd
import hydra
import wandb


def main_mimo(cfg : dict, rep : int, seed : int) -> None:
    config = cfg.experiments["hyperparameters"]

    set_seed(seed)

    #Select model to train
    is_resnet = config.is_resnet
    model_name = config.model_name + 'Wide' if is_resnet else config.model_name	
    naive = config.is_naive
    n_subnetworks = config.n_subnetworks
    dataset = config.dataset
    plot = config.plot

    # make relevant dirs
    make_dirs(f"models/classification/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"models/classification/checkpoints/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"reports/figures/losses/classification/{model_name}/{dataset}/M{n_subnetworks}/")

    # model parameters
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    weight_decay = compute_weight_decay(config.sigma1)
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs

    if naive:
        if is_resnet:
            depth = config.depth
            widen_factor = config.widen_factor
            p = config.dropout_rate
            print(f"Training Naive WideResnet({depth}, {widen_factor}) model with {n_subnetworks} subnetworks on classification task.")
            model_name = f'C_NaiveWide/{dataset}/M{n_subnetworks}/' + 'C_NaiveWide' + f'_{depth}_{widen_factor}_{n_subnetworks}_members_rep{rep}'
        else:
            print(f"Training Naive model with {n_subnetworks} subnetworks on classification task.")
            model_name = f"C_Naive/{dataset}/M{n_subnetworks}/" + config.model_name + f'_{n_subnetworks}_members_rep{rep}'

    else:
        if is_resnet:
            depth = config.depth
            widen_factor = config.widen_factor
            p = config.dropout_rate
            print(f"Training MIMO WideResnet({depth}, {widen_factor}) model with {n_subnetworks} subnetworks on classification task.")
            model_name = f"C_MIMOWide/{dataset}/M{n_subnetworks}/" + 'C_MIMOWide' + f'_{depth}_{widen_factor}_{n_subnetworks}_members_rep{rep}'
        elif n_subnetworks == 1:
            print(f"Training baseline model on classification task.")
            model_name = f"C_MIMO/{dataset}/M{n_subnetworks}/" + config.model_name + f"_rep{rep}"
        else:
            print(f"Training MIMO model with {n_subnetworks} subnetworks on classification task.")
            model_name = f"C_MIMO/{dataset}/M{n_subnetworks}/" + config.model_name + f'_{n_subnetworks}_members_rep{rep}'
    
    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar100("data/", data_aug=True) if dataset == 'CIFAR100' else load_cifar10("data/", data_aug=True)
    n_classes = 100 if dataset == 'CIFAR100' else 10
    hidden_units1 = config.n_hidden_units
    channels1 = config.channels1
    channels2 = config.channels2
    channels3 = config.channels3
    batch_repetition = config.batch_repetition
    gamma = config.gamma

    if naive == False:
        trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, shuffle=True, collate_fn=lambda x: C_train_collate_fn(x, n_subnetworks, batch_repetition), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, n_subnetworks), drop_last=False)
        # model = MIMOWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p, n_classes=n_classes) if is_resnet else C_MIMONetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
        model = MIMOWideResnet(depth=depth, widen_factor=widen_factor, dropRate=p, n_classes=n_classes, n_subnetworks=n_subnetworks) if is_resnet else C_MIMONetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
    else:
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, collate_fn=lambda x: C_Naive_train_collate_fn(x, n_subnetworks), drop_last=True, worker_init_fn=seed_worker, generator=g)
        valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, n_subnetworks), drop_last=False)
        # model = NaiveWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p, n_classes=n_classes) if is_resnet else C_NaiveNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
        model = NaiveWideResnet(depth=depth, widen_factor=widen_factor, dropRate=p, n_classes=n_classes, n_subnetworks=n_subnetworks) if is_resnet else C_NaiveNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
        
    # model.apply(init_weights)
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=gamma)


    if is_resnet:
        run_MIMO_resnet(model, optimizer, scheduler, trainloader, valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)
    # else:
    # train_classification(model, optimizer, scheduler, trainloader, valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)
    # if plot==True:
    #     plot_loss(losses, val_losses, model_name=model_name, task='classification')

def main_bnn(cfg : dict, rep : int, seed : int) -> None:
    config = cfg.experiments["hyperparameters"]

    set_seed(seed)

    #Select model to train
    is_resnet = config.is_resnet
    model_name = config.model_name + 'Wide' if is_resnet else config.model_name
    dataset = config.dataset
    plot = config.plot

    # make relevant dirs
    make_dirs(f"models/classification/{model_name}/{dataset}/")
    make_dirs(f"models/classification/checkpoints/{model_name}/{dataset}/")
    make_dirs(f"reports/figures/losses/classification/{model_name}/{dataset}/")

    # model parameters
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs
    pi = config.pi
    # sigma1 = torch.exp(torch.tensor(config.sigma1))
    # sigma2 = torch.exp(torch.tensor(config.sigma2))
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)
    hidden_units1 = config.n_hidden_units
    channels1 = config.channels1
    channels2 = config.channels2
    channels3 = config.channels3


    #Select model to train
    if is_resnet:
        depth = config.depth
        widen_factor = config.widen_factor
        p = config.dropout_rate
        model_name = f"C_BNNWide/{dataset}/" + "C_BNNWide" + f"_{depth}_{widen_factor}_rep{rep}"
        print(f"Training BNN WideResnet({depth}, {widen_factor}) model on classification task.")
    else:
        model_name = f"C_BNN/{dataset}/" + config.model_name + f"_rep{rep}" 
        print(f"Training BNN model on classification task.")

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar100("data/") if dataset == 'CIFAR100' else load_cifar10("data/")
    n_classes = 100 if dataset == 'CIFAR100' else 10
    CIFAR_trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    CIFAR_valloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    
    # BNN_model = BayesianWideResNet(depth=depth, widen_factor=widen_factor, dropRate=p, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else BayesianConvNeuralNetwork(hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    BNN_model = BayesianWideResnet(depth, widen_factor, p, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else BayesianConvNeuralNetwork(hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    BNN_model = BNN_model.to(device)
    # optimizer = torch.optim.Adam(BNN_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(BNN_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    if is_resnet:
        run_BNN_resnet(BNN_model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)
    # else:
    # losses, log_priors, log_variational_posteriors, NLLs, val_losses = train_BNN_classification(BNN_model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)

    # if plot == True:
    #     plot_loss(losses, val_losses, model_name=model_name, task='classification')
    #     plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name=model_name, task='classification')

def main_mimbo(cfg : dict, rep : int, seed : int) -> None:
    config = cfg.experiments["hyperparameters"]

    set_seed(seed)

    #Select model to train
    is_resnet = config.is_resnet
    model_name = config.model_name + 'Wide' if is_resnet else config.model_name
    dataset = config.dataset
    plot = config.plot
    n_subnetworks = config.n_subnetworks

    # make relevant dirs
    make_dirs(f"models/classification/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"models/classification/checkpoints/{model_name}/{dataset}/M{n_subnetworks}/")
    make_dirs(f"reports/figures/losses/classification/{model_name}/{dataset}/M{n_subnetworks}/")

    # model parameters
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    train_epochs = config.train_epochs
    val_every_n_epochs = config.val_every_n_epochs
    pi = config.pi
    # sigma1 = torch.exp(torch.tensor(config.sigma1))
    # sigma2 = torch.exp(torch.tensor(config.sigma2))
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)
    hidden_units1 = config.n_hidden_units
    channels1 = config.channels1
    channels2 = config.channels2
    channels3 = config.channels3
    batch_repetition = config.batch_repetition
    gamma = config.gamma

    if is_resnet:
        depth = config.depth
        widen_factor = config.widen_factor
        p = config.dropout_rate
        model_name = f"C_MIMBOWide/{dataset}/M{n_subnetworks}/" + "C_MIMBOWide" + f"_{depth}_{widen_factor}_{n_subnetworks}_members_rep{rep}"
        print(f"Training MIMBO WideResnet({depth}, {widen_factor}) model with {n_subnetworks} subnetworks on classification task.")
    else:
        model_name = f"C_MIMBO/{dataset}/M{n_subnetworks}/" + config.model_name + f"_rep{rep}"
        print(f"Training MIMBO model with {n_subnetworks} subnetworks on classification task.")

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traindata, valdata, _ = load_cifar100("data/") if dataset == 'CIFAR100' else load_cifar10("data/")
    n_classes = 100 if dataset == 'CIFAR100' else 10
    CIFAR_trainloader = DataLoader(traindata, batch_size=batch_size*n_subnetworks, collate_fn=lambda x: C_train_collate_fn(x, n_subnetworks, batch_repetition), shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    CIFAR_valloader = DataLoader(valdata, batch_size=batch_size, collate_fn=lambda x: C_test_collate_fn(x, n_subnetworks), shuffle=False, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)

    # MIMBO_model =  MIMBOWideResnet(n_subnetworks, depth, widen_factor, p, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else MIMBOConvNeuralNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    MIMBO_model = MIMBOWideResnet(depth=depth, widen_factor=widen_factor, dropRate=p, n_classes=n_classes, n_subnetworks=n_subnetworks, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else MIMBOConvNeuralNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    MIMBO_model = MIMBO_model.to(device)
    # optimizer = torch.optim.Adam(MIMBO_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(MIMBO_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    if is_resnet:
        run_BNN_resnet(MIMBO_model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)
    # else:
    # train_BNN_classification(MIMBO_model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=train_epochs, model_name=model_name, val_every_n_epochs=val_every_n_epochs, checkpoint_every_n_epochs=2, device=device)

    # if plot == True:
    #     plot_loss(losses, val_losses, model_name=model_name, task='classification')
    #     plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name=model_name, task='classification')


@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:
    config = cfg.experiments["hyperparameters"]

    reps = config.repetitions

    # repeat experiments 5 times
    for r in range(1,reps+1):

        seed = config.seed + r - 1

        mode = config.mode
        is_resnet = config.is_resnet
        if config.model_name == 'C_BNN':
            if is_resnet:
                depth = config.depth
                widen_factor = config.widen_factor
                name = f"{config.model_name}_{depth}_{widen_factor}_classification_rep{r}"
            else:
                name = f"{config.model_name}_classification_rep{r}"
        else:
            if is_resnet:
                depth = config.depth
                widen_factor = config.widen_factor
                name = f"{config.model_name}_{depth}_{widen_factor}_{config.n_subnetworks}_members_classification_rep{r}"
            else:
                name = f"{config.model_name}_{config.n_subnetworks}_members_classification_rep{r}"
        
        wandb.init(
            project="FinalRuns", 
            name=name,
            # mode='disabled',
            # name="DELETE_THIS", 
            config = omegaconf.OmegaConf.to_container(cfg),
            group=config.dataset)


        print(f"Running experiment {r} of {reps} using seed {seed}")
        match mode:
            case 0: #baseline
                cfg.experiments["hyperparameters"].n_subnetworks = 1
                main_mimo(cfg, rep=r, seed=seed)
            case 1: #MIMO
                main_mimo(cfg, rep=r, seed=seed)
            case 2: #Naive multi-headed
                cfg.experiments["hyperparameters"].is_naive = True 
                main_mimo(cfg, rep=r, seed=seed)
            case 3: #BNN
                main_bnn(cfg, rep=r, seed=seed)
            case 4: # MIMBO
                main_mimbo(cfg, rep=r, seed=seed)

        wandb.finish()

if __name__ == "__main__":
    main()