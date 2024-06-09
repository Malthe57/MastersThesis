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
from data.make_dataset import make_toydata, make_multidim_toydata
from data.OneD_dataset import train_collate_fn, test_collate_fn, naive_collate_fn, bnn_collate_fn, load_toydata
from training_loops import train_var_regression, train_BNN
from utils.utils import set_seed, seed_worker, get_zero_mean_mixture_variance, compute_weight_decay


def prepare_sweep_dict(model_name: str, dataset: str, n_subnetworks : int, batch_size: int, n_hidden_units: int, n_hidden_units2: int, lr : float):

    sweep_config = {
            "name": f"regression_{model_name}_{dataset}_{n_subnetworks}",
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

        'optimizer': {
            'values': ['Adam']
        },

        # 'weight_decay': {
        #     'values': [None]
        # },

        'sigma1': {
            'values': [1, 3, 5, 10, 30, 50, 100, 5000]
        },

        'sigma2': {
            'values': [0]
        },

        'pi': {
            'values': [1.0]
        },

        'lr': {
            'values': [lr]
        }
    }
    # if 'C_BNN' in model_name or 'C_MIMBO' in model_name:
    #     parameters_dict.update({
    #         'sigma1': {
    #             'values': [0.1, 0.5, 1, 3, 5, 7.5, 10]
    #     }})
        # parameters_dict.update({
        #     'sigma2': {
        #         'values': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        # }})
    # elif 'C_MIMO' in model_name or 'C_Naive' in model_name:
    #     parameters_dict.update({
    #         'weight_decay': {
    #             'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    #     }})

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
        traindata, valdata, _, input_dim, _, = load_toydata(normalise=True)

    elif config.dataset=='multitoydata':
        make_multidim_toydata()
        traindata, valdata, _, input_dim, _, _, _ = load_multireg_data(config.dataset)
    
    elif config.dataset=='crimedata':
        prepare_crime()
        traindata, valdata, _, input_dim, _, _, _  = load_multireg_data(config.dataset)

    if "MIMO" in name or 'Baseline' in name:
        train_collate_fn_ = lambda x: train_collate_fn(x, n_subnetworks)
        val_collate_fn_ = lambda x: test_collate_fn(x, n_subnetworks)
    elif 'Naive' in name:
        train_collate_fn_ = lambda x: naive_collate_fn(x, n_subnetworks)
        val_collate_fn_ = lambda x: naive_collate_fn(x, n_subnetworks)
    elif 'BNN' in name:
        train_collate_fn_ = bnn_collate_fn
        val_collate_fn_ = bnn_collate_fn
    elif 'MIMBO' in name:
        train_collate_fn_ = lambda x: train_collate_fn(x, n_subnetworks)
        val_collate_fn_ = lambda x: test_collate_fn(x, n_subnetworks)

    trainloader = DataLoader(traindata, batch_size=config.batch_size*n_subnetworks, shuffle=True, collate_fn=train_collate_fn_, drop_last=True, pin_memory=True)
    valloader = DataLoader(valdata, batch_size=config.batch_size, shuffle=False, collate_fn=val_collate_fn_, drop_last=False, pin_memory=True)

    return trainloader, valloader, input_dim

def get_model(config, input_dim, device):

    name = config.name
    n_hidden_units = config.n_hidden_units
    n_hidden_units2 = config.n_hidden_units2
    n_subnetworks = config.n_subnetworks
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)
    pi = config.pi


    if 'MIMO' in name or 'Baseline' in name:
        model = VarMIMONetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)
    elif 'Naive' in name:
        model = VarNaiveNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, input_dim=input_dim)
    elif 'BNN' in name:
        model = BayesianNeuralNetwork(n_hidden_units, n_hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, input_dim=input_dim, device=device)
    elif 'MIMBO' in name:
        model = MIMBONeuralNetwork(n_subnetworks, n_hidden_units, n_hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, input_dim=input_dim, device=device)
    return model

def get_optimizer(model, config):

    if config.name == 'MIMO' or config.name == 'Naive' or config.name == 'Baseline':
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=compute_weight_decay(config.sigma1))
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=compute_weight_decay(config.sigma1), momentum=0.9)

    elif config.name == 'BNN' or config.name == 'MIMBO':
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    return optimizer

def train(config=None):

    run = wandb.init(config=config)
    config = wandb.config

    run.name = f"{config.name}_{config.n_subnetworks}subnetworks_{config.dataset}_sigma1_{config.sigma1}"

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    trainloader, valloader, input_dim = get_dataloaders(config)
    model = get_model(config, input_dim, device=device)
    model = model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    if 'BNN' in config.name or 'MIMBO' in config.name:
        train_BNN(model, optimizer, scheduler, trainloader, valloader, epochs=1500, model_name=config.name, val_every_n_epochs=1, device=device)
    else:
        train_var_regression(model, optimizer, scheduler, trainloader, valloader, epochs=1500, model_name=config.name, val_every_n_epochs=1, device=device)

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
    lr = config.learning_rate

    sweep_config = prepare_sweep_dict(model_name, dataset, n_subnetworks, batch_size, n_hidden_units, n_hidden_units2, lr)

    sweep_id = wandb.sweep(sweep_config, project="RegressionSweeps")

    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()




        

            
        



