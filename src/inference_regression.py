import torch
import numpy as np
from torch.utils.data import DataLoader
from data.OneD_dataset import test_collate_fn, naive_collate_fn, generate_data, ToyDataset, bnn_collate_fn, load_toydata
from data.MultiD_dataset import MultiDataset, load_multireg_data
from visualization.visualize_mimo import plot_regression
from utils.utils import make_dirs
import glob
import os
import pandas as pd
from models.mimo import C_MIMONetwork, C_NaiveNetwork, MIMONetwork, NaiveNetwork
import argparse

def inference(model, testloader):
    predictions = []
    pred_individual = []
    

    for test_x, test_y in testloader:
        output, individual_outputs = model(test_x.float())
        
        predictions.extend(list(output.detach().numpy()))
        pred_individual.extend(list(individual_outputs.detach().numpy()))

    return np.array(predictions), np.array(pred_individual), 

def var_inference(model, testloader, device='cpu'):
    mu_list = []
    sigma_list = []
    mus_list = []
    sigmas_list = []

    for test_x, test_y in testloader:
        test_x = test_x.float().to(device)
        mu, sigma, mus, sigmas = model(test_x.float())
        
        mu_list.extend(list(mu.detach().numpy()))
        sigma_list.extend(list(sigma.detach().numpy()))
        mus_list.extend(list(mus.detach().numpy()))
        sigmas_list.extend(list(sigmas.detach().numpy()))

    return np.array(mu_list), np.array(sigma_list), np.array(mus_list), np.array(sigmas_list)

def mimbo_inference(model, testloader, device='cpu'):
    predictions = []
    stds = []
    mus_list = []
    sigmas_list = []

    for x_test, y_test in testloader:
        x_test, y_test = x_test.float().to(device), y_test.float()
        with torch.no_grad():
            mu, sigma, mus, sigmas = model.inference(x_test, sample=True)
            predictions.append(mu.cpu().detach().numpy())
            stds.append(sigma.cpu().detach().numpy())
            mus_list.append(mus.cpu().detach().numpy())
            sigmas_list.append(sigmas.cpu().detach().numpy())

    return np.array(predictions), np.array(stds), np.array(mus), np.array(sigmas)

# get predictions and individual predictions for MIMO and Naive models
def get_mimo_predictions(model_path, Ms, testdata, N_test=500):

    predictions_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        predictions, pred_individual = inference(model, testloader)

        predictions_matrix[i, :] = predictions
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list

def get_naive_predictions(model_path, Ms, testdata, N_test=500):

    predictions_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        predictions, pred_individual = inference(model, testloader)

        predictions_matrix[i, :] = predictions
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list

def get_var_mimo_predictions(model_path, Ms, testdata, N_test=500, reps=1):
    """
    inputs:
        model_path: list of paths to the models
        Ms: list of number of subnetworks
        N_test: number of test samples
    outputs:
        mu_matrix: matrix of shape (len(model_path), N_test) containing the mean predictions of each ensemble with M subnetworks
        sigma_matrix: matrix of shape (len(model_path), N_test) containing the standard deviation predictions of each ensemble with M subnetworks
        mu_individual_list: array of shape (N_test, sum(Ms)) containing the individual mean predictions of each subnetwork
        sigma_individual_list: array of shape (N_test, sum(Ms)) containing the individual standard deviation predictions of each subnetwork

        Note: if Ms = [1,2,3,4,5] then mu_individual_list is (N_test, 15). 
        Which means:
            mu_individual_list[:,0] is the baseline
            mu_individual_list[:,1:3] is the MIMO with M = 2
            mu_individual_list[:,3:6] is the MIMO with M = 3
            mu_individual_list[:,6:10] is the MIMO with M = 4
            mu_individual_list[:,10:15] is the MIMO with M = 5
    """

    mu_matrix = np.zeros((reps, len(model_path), N_test))
    sigma_matrix = np.zeros((reps, len(model_path), N_test))
    mu_individual_lists = []
    sigma_individual_lists = []

    testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)
    for i, path in enumerate(model_path):
        M = Ms[i]

        mu_individual_list = []
        sigma_individual_list = []
        for j, model in enumerate(path):
                
            model = torch.load(model)
            mu, sigma, mus, sigmas = var_inference(model, testloader)

            mu_matrix[j, i, :] = mu
            sigma_matrix[j, i, :] = sigma
            mu_individual_list.append(mus)
            sigma_individual_list.append(sigmas)
        mu_individual_lists.append(mu_individual_list)
        sigma_individual_lists.append(sigma_individual_list)
            
    return mu_matrix, sigma_matrix, np.concatenate(mu_individual_lists, axis=2), np.concatenate(sigma_individual_lists, axis=2)
    

def get_var_naive_predictions(model_path, Ms, testdata, N_test=500, reps=1):

    mu_matrix = np.zeros((reps, len(model_path), N_test))
    sigma_matrix = np.zeros((reps, len(model_path), N_test))
    mu_individual_lists = []
    sigma_individual_lists = []

    testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)
    for i, path in enumerate(model_path):
        M = Ms[i]

        mu_individual_list = []
        sigma_individual_list = []
        for j, model in enumerate(path):
                
            model = torch.load(model)
            mu, sigma, mus, sigmas = var_inference(model, testloader)

            mu_matrix[j, i, :] = mu
            sigma_matrix[j, i, :] = sigma
            mu_individual_list.append(mus)
            sigma_individual_list.append(sigmas)
        mu_individual_lists.append(mu_individual_list)
        sigma_individual_lists.append(sigma_individual_list)
            
    return mu_matrix, sigma_matrix, np.concatenate(mu_individual_lists, axis=2), np.concatenate(sigma_individual_lists, axis=2)

def get_bnn_predictions(bnn_path, testdata, N_test=500, reps=1):

    mu_matrix = np.zeros((reps, 1, N_test))
    sigma_matrix = np.zeros((reps, 1, N_test))

    testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=bnn_collate_fn, pin_memory=True)

    for i, path in enumerate(bnn_path):
        model = torch.load(path)
        for x_test, y_test in testloader:
            x_test, y_test = x_test.float(), y_test.float()
            with torch.no_grad():
                mu, sigma = model.inference(x_test, sample=True, n_samples=10)
                mu_matrix[i,:,:] = mu.cpu().detach().numpy()
                sigma_matrix[i,:,:] = sigma.cpu().detach().numpy()

    return mu_matrix, sigma_matrix

def get_mimbo_predictions(model_path, Ms, testdata, N_test=500, reps=1):
    mu_matrix = np.zeros((reps, len(model_path), N_test))
    sigma_matrix = np.zeros((reps, len(model_path), N_test))
    mu_individual_lists = []
    sigma_individual_lists = []

    
    for i, path in enumerate(model_path):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)
        mu_individual_list = []
        sigma_individual_list = []
        for j, model in enumerate(path):
                
            model = torch.load(model)
            mu, sigma, mus, sigmas = mimbo_inference(model, testloader)

            mu_matrix[j, i, :] = mu
            sigma_matrix[j, i, :] = sigma
            mu_individual_list.append(mus.T)
            sigma_individual_list.append(sigmas.T)
        mu_individual_lists.append(mu_individual_list)
        sigma_individual_lists.append(sigma_individual_list)
            
    return mu_matrix, sigma_matrix, np.concatenate(mu_individual_lists, axis=2), np.concatenate(sigma_individual_lists, axis=2)

def main(model_name, model_path, Ms, dataset_path, reps):

    if dataset_path[5]=='t':
        _, _, testdata, _, test_length = load_toydata(normalise=True)
        dataset = 'toydata'
    elif dataset_path[5]=='m':
        if dataset_path[18]=='m':
            dataset = 'multitoydata'

        elif dataset_path[18]=='n':
            dataset = 'newsdata'
            
        elif dataset_path[18]=='c':
            dataset = 'crimedata'
        
        _, _, testdata, _, test_length, _, _ = load_multireg_data(dataset, standardise=True)

    match model_name:
        case "Baseline":
            make_dirs(f"reports/Logs/MIMO/{dataset}/")
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_var_mimo_predictions(model_path, Ms, testdata, N_test=test_length, reps=reps)
            np.savez(f'reports/Logs/MIMO/{dataset}/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)
        case "MIMO":
            make_dirs(f"reports/Logs/MIMO/{dataset}/")
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_var_mimo_predictions(model_path, Ms, testdata, N_test=test_length, reps=reps)
            np.savez(f'reports/Logs/MIMO/{dataset}/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)
        case "Naive":
            make_dirs(f"reports/Logs/Naive/{dataset}/")
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_var_naive_predictions(model_path, Ms, testdata, N_test=test_length, reps=reps)
            np.savez(f'reports/Logs/Naive/{dataset}/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)
        case "BNN":
            make_dirs(f"reports/Logs/BNN/{dataset}/")
            mu_matrix, sigma_matrix = get_bnn_predictions(model_path, testdata, N_test=test_length, reps=reps)
            np.savez(f'reports/Logs/BNN/{dataset}/{model_name}', predictions = mu_matrix, mu_individual = [], predicted_std = sigma_matrix, sigma_individual = [])
        case "MIMBO":
            make_dirs(f"reports/Logs/MIMBO/{dataset}/")
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_mimbo_predictions(model_path, Ms, testdata, N_test=test_length, reps=reps)
            np.savez(f'reports/Logs/MIMBO/{dataset}/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='MIMBO', help='Model name [Baseline, MIMO, Naive, BNN, MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    parser.add_argument('--dataset', type=str, default='multitoydata', help='Dataset in use:\n Regression: [1D, newsdata, crimedata]\n Classification: [cifar10, cifar100]')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions - should match the number of models in folder')
    args = parser.parse_args()

    if type(args.Ms) == list:
        args.Ms = args.Ms[0]
    Ms = [int(M) for M in args.Ms.split(',')]
    base_path = f'models/regression/{args.model_name}/{args.dataset}/'
    M_path = [os.path.join(base_path, f"M{M}") for M in Ms]
    if args.model_name == "MIMO" or args.model_name == "Naive" or args.model_name == "MIMBO":
        M_path = [os.path.join(base_path, f"M{M}") for M in Ms]
        model_paths = [[os.path.join(p, model) for model in os.listdir(p)] for p in M_path]
    else:
        model_paths = [os.path.join(base_path, model) for model in os.listdir(base_path)]
    

    if args.dataset=='1D':
        dataset_path = 'data/toydata/test_data.csv'
    else:
        dataset_path = f'data/multidimdata/{args.dataset}/{args.dataset[:-4]}_test_data.csv'

    main(args.model_name, model_paths, Ms, dataset_path, args.reps)
    print("Done")

