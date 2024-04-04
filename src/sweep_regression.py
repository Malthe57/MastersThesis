import numpy as np
import torch
import hydra
import omegaconf
import wandb
from torch.utils.data import DataLoader 
from models.mimo import VarMIMONetwork, VarNaiveNetwork
from models.bnn import BayesianNeuralNetwork
from models.mimbo import MIMBONeuralNetwork
from data.MultiD_dataset import prepare_news, prepare_crime, load_multireg_data
from data.make_dataset import make_toydata
from data.OneD_dataset import train_collate_fn, test_collate_fn, naive_collate_fn, bnn_collate_fn, load_toydata
from training_loops import train_var_regression, train_BNN
from utils.utils import set_seed, seed_worker, get_zero_mean_mixture_variance, compute_weight_decay


def prepare_sweep_dict(model_name: str, dataset: str, n_subnetworks : int, batch_size: int, n_hidden_units: int, n_hidden_units2: int):

    sweep_config = {
            "method": "grid",
            }

    metric = {
            'name': 'Val loss',
            'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    parameters_dict = {

        "name" : {
            "values": [model_name]
        },

        'dataset': {
            'values': [dataset]
        },

        'batch_size': {
            'values': [batch_size]
        },

        'n_subnetworks': {
            'values': [n_subnetworks]
        },

        'n_hidden_units': {
            'values': [n_hidden_units]
        },

        'n_hidden_units2': {
            'values': [n_hidden_units2]
        },

        'pi': {
            'values': [0.5]
        },

        'sigma1': {
            'values': [0.01, 0.1, 1, 10, 50]
        },
        'sigma2': {
            'values': [0.01, 0.1, 1, 10, 50]
        },

        'lr': {
            'values': [3e-4]
        }
    }
    sweep_config['parameters'] = parameters_dict
    
    return sweep_config

def get_dataloaders(config : dict):

    name = config.name
    n_subnetworks = config.n_subnetworks

    #Set generator seed
    g = torch.Generator()
    g.manual_seed(0)

    if config.dataset=="1D":
        make_toydata()
        traindata, valdata, _, input_dim, _ = load_toydata(normalise=True)

    elif config.dataset=="newsdata":
        prepare_news()
        traindata, valdata, _, input_dim, _ = load_multireg_data(config.dataset)
    
    elif config.dataset=='crimedata':
        prepare_crime()
        traindata, valdata, _, input_dim, _  = load_multireg_data(config.dataset)

    if "C_MIMO" in name:
        train_collate_fn = lambda x: train_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: test_collate_fn(x, n_subnetworks)
    elif 'C_Naive' in name:
        train_collate_fn = lambda x: naive_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: naive_collate_fn(x, n_subnetworks)
    elif 'C_BNN' in name:
        train_collate_fn = bnn_collate_fn
        val_collate_fn = bnn_collate_fn
    elif 'C_MIMBO' in name:
        train_collate_fn = lambda x: train_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: test_collate_fn(x, n_subnetworks)

    trainloader = DataLoader(traindata, batch_size=config.batch_size*n_subnetworks, shuffle=True, collate_fn=train_collate_fn, drop_last=True, pin_memory=True)
    valloader = DataLoader(valdata, batch_size=config.batch_size, shuffle=False, collate_fn=val_collate_fn, drop_last=False, pin_memory=True)

    return trainloader, valloader, input_dim

def get_model(config, input_dim, device):

    name = config.name
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    n_subnetworks = config.n_subnetworks
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)
    pi = config.pi

    if 'C_MIMO' in name:
        model = VarMIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)
    elif 'C_Naive' in name:
        model = VarNaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)
    elif 'C_BNN' in name:
        model = BayesianNeuralNetwork(n_hidden_units, n_hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, input_dim=input_dim)
    elif 'C_MIMBO' in name:
        model = MIMBONeuralNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, input_dim=input_dim)
    return model


@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:

    config = cfg.experiments["hyperparameters"]

    set_seed(1871)

    dataset = config.dataset
    is_resnet = config.is_resnet
    n_subnetworks = config.n_subnetworks
    batch_size = config.batch_size
    model_name = config.model_name
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    if is_resnet:
        model_name += "Wide"

    sweep_config = prepare_sweep_dict(model_name, dataset, n_subnetworks, batch_size, n_hidden_units, n_hidden_units2)

    sweep_id = wandb.sweep(sweep_config, project="MastersThesis")

    wandb.agent(sweep_id, function=train)

def train(config=None):

    run = wandb.init(config=config)
    config = wandb.config

    run.name = f"regression_{config.name}_{config.dataset}_{config.n_subnetworks}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, valloader, input_dim = get_dataloaders(config)
    model = get_model(config, input_dim, device=device)
    model = model.to(device)

    if 'C_BNN' in config.name or 'C_MIMBO' in config.name:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        mixture_var = get_zero_mean_mixture_variance(config.sigma1, config.sigma2, config.pi)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=compute_weight_decay(mixture_var))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    if 'C_BNN' in config.name or 'C_MIMBO' in config.name:
        train_BNN(model, optimizer, scheduler, trainloader, valloader, 300, model_name=config.name, val_every_n_epochs=1)
    else:
        train_var_regression(model, optimizer, scheduler, trainloader, valloader, 300, model_name=config.name, val_every_n_epochs=1)

if __name__ == "__main__":
    main()




        

            
        



