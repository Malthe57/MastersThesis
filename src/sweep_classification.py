import numpy as np
import torch
import hydra
import omegaconf
import wandb
from torch.utils.data import DataLoader 
from models.mimo import C_MIMONetwork, C_NaiveNetwork, MIMOWideResnet, NaiveWideResnet
from models.bnn import BayesianConvNeuralNetwork, BayesianWideResnet
from models.mimbo import MIMBOConvNeuralNetwork, MIMBOWideResnet
from training_loops import train_classification, train_BNN_classification
from data.CIFAR10 import load_cifar10, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.CIFAR100 import load_cifar100
from utils.utils import set_seed, seed_worker, get_zero_mean_mixture_variance, compute_weight_decay


def prepare_sweep_dict(model_name: str, dataset: str, is_resnet: bool, n_subnetworks : int, batch_size: int, n_hidden_units, channels1, channels2, channels3):

    sweep_config = {
            "name": f"classification_{model_name}_{dataset}_subnetworks{n_subnetworks}",
            "method": "grid",
            }

    metric = {
            'name': 'Val accuracy',
            'goal': 'maximize'
        }

    sweep_config['metric'] = metric

    parameters_dict = {

        "name" : {
            "values": [model_name]
        },

        'dataset': {
            'values': [dataset]
        },
        'is_resnet': {
            'values': [is_resnet]
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

        'channels1': {
            'values': [channels1]
        },

        'channels2': {
            'values': [channels2]
        },

        'channels3': {
            'values': [channels3]
        },

        'optimizer': {
            'values': ['Adam']
        },

        'depth': {
            'values': [28]
        },

        'widen_factor': {
            'values': [10]
        },

        'dropout_rate': {
            'values': [0.0, 0.15, 0.3]
        },
        
        'weight_decay': {
            'values': [None]
        },

        'sigma1': {
            'values': [1, 3, 5, 10, 30, 50, 5000]
        },

        'sigma2': {
            'values': [0.0]
        },

        'pi': {
            'values': [1.0]
        },

        'lr': {
            'values': [1e-3]
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

    traindata, valdata, _ = load_cifar100("data/") if config.dataset == 'CIFAR100' else load_cifar10("data/")

    if "C_MIMO" in name:
        train_collate_fn = lambda x: C_train_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: C_test_collate_fn(x, n_subnetworks)
    elif 'C_Naive' in name:
        train_collate_fn = lambda x: C_Naive_train_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: C_Naive_test_collate_fn(x, n_subnetworks)
    elif 'C_BNN' in name:
        train_collate_fn = None
        val_collate_fn = None
    elif 'C_MIMBO' in name:
        train_collate_fn = lambda x: C_train_collate_fn(x, n_subnetworks)
        val_collate_fn = lambda x: C_test_collate_fn(x, n_subnetworks)

    CIFAR_trainloader = DataLoader(traindata, batch_size=config.batch_size*n_subnetworks, collate_fn=train_collate_fn, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    CIFAR_valloader = DataLoader(valdata, batch_size=config.batch_size, collate_fn=val_collate_fn, shuffle=True, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g)

    return CIFAR_trainloader, CIFAR_valloader

def get_model(config, device):

    name = config.name
    is_resnet = config.is_resnet
    n_classes = 100 if config.dataset == 'CIFAR100' else 10
    n_subnetworks = config.n_subnetworks
    sigma1 = torch.tensor(config.sigma1)
    sigma2 = torch.tensor(config.sigma2)
    pi = config.pi
    depth = config.depth
    widen_factor = config.widen_factor
    p = config.dropout_rate
    hidden_units1 = config.n_hidden_units
    channels1 = config.channels1
    channels2 = config.channels2
    channels3 = config.channels3

    if 'C_MIMO' in name:
        model = MIMOWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p, n_classes=n_classes) if is_resnet else C_MIMONetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
    elif 'C_Naive' in name:
        model = NaiveWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p, n_classes=n_classes) if is_resnet else C_NaiveNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes)
    elif 'C_BNN' in name:
        model = BayesianWideResnet(depth, widen_factor, p, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else BayesianConvNeuralNetwork(hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    elif 'C_MIMBO' in name:
        model = MIMBOWideResnet(n_subnetworks=n_subnetworks, depth=depth, widen_factor=widen_factor, dropout_rate=p, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) if is_resnet else MIMBOConvNeuralNetwork(n_subnetworks=n_subnetworks, hidden_units1=hidden_units1, channels1=channels1, channels2=channels2, channels3=channels3, n_classes=n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
    
    return model

def get_optimizer(model, config):

    if 'C_MIMO' in config.name or 'C_Naive' in config.name:
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=compute_weight_decay(config.sigma1))
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=compute_weight_decay(config.sigma1), momentum=0.9, nesterov=True)

    elif 'C_BNN' in config.name or 'C_MIMBO' in config.name:
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, nesterov=True)

    return optimizer

def train(config=None):

    run = wandb.init(config=config)
    config = wandb.config

    run.name = f"{config.name}_{config.dataset}_sigma1_{config.sigma1}_dropout_{config.dropout_rate}_subnetworks_{config.n_subnetworks}"
    if config.is_resnet:
        run.name += f"_resnet"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CIFAR_trainloader, CIFAR_valloader = get_dataloaders(config)
    model = get_model(config, device=device)
    model = model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    if 'C_BNN' in config.name or 'C_MIMBO' in config.name:
        train_BNN_classification(model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=40, model_name=config.name, val_every_n_epochs=1, checkpoint_every_n_epochs=100, device=device, save=False)
    else:
        train_classification(model, optimizer, scheduler, CIFAR_trainloader, CIFAR_valloader, epochs=40, model_name=config.name, val_every_n_epochs=1, checkpoint_every_n_epochs=100, device=device, save=False)


@hydra.main(config_path="../conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: dict) -> None:

    config = cfg.experiments["hyperparameters"]

    set_seed(config.seed)

    dataset = config.dataset
    is_resnet = config.is_resnet
    n_subnetworks = config.n_subnetworks
    batch_size = config.batch_size
    model_name = config.model_name
    n_hidden_units = config.n_hidden_units
    channels1 = config.channels1
    channels2 = config.channels2
    channels3 = config.channels3

    if is_resnet:
        model_name += "Wide"

    sweep_config = prepare_sweep_dict(model_name, dataset, is_resnet, n_subnetworks, batch_size, n_hidden_units, channels1, channels2, channels3)

    sweep_id = wandb.sweep(sweep_config, project="ClassificationSweeps")

    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()




        

            
        



