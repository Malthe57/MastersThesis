import torch
import numpy as np
from torch.utils.data import DataLoader
from data.OneD_dataset import test_collate_fn, naive_collate_fn, generate_data, ToyDataset, bnn_collate_fn
from data.CIFAR10 import C_Naive_test_collate_fn, C_Naive_train_collate_fn, C_test_collate_fn, C_train_collate_fn, load_cifar
from visualization.visualize_mimo import plot_regression
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

def var_inference(model, testloader):
    mu_list = []
    sigma_list = []
    mus_list = []
    sigmas_list = []

    for test_x, test_y in testloader:
        mu, sigma, mus, sigmas = model(test_x.float())
        
        mu_list.extend(list(mu.detach().numpy()))
        sigma_list.extend(list(sigma.detach().numpy()))
        mus_list.extend(list(mus.detach().numpy()))
        sigmas_list.extend(list(sigmas.detach().numpy()))

    return np.array(mu_list), np.array(sigma_list), np.array(mus_list), np.array(sigmas_list)

# get predictions and individual predictions for MIMO and Naive models
def get_mimo_predictions(model_path, Ms, testdata, N_test=200):

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

def get_naive_predictions(model_path, Ms, testdata, N_test=200):

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

def get_var_mimo_predictions(model_path, Ms, testdata, N_test=200):

    mu_matrix = np.zeros((len(model_path), N_test))
    sigma_matrix = np.zeros((len(model_path), N_test))
    mu_individual_list = []
    sigma_individual_list = []


    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        mu, sigma, mus, sigmas = var_inference(model, testloader)

        mu_matrix[i, :] = mu
        sigma_matrix[i, :] = sigma
        mu_individual_list.append(mus)
        sigma_individual_list.append(sigmas)
            
    return mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list

def get_var_naive_predictions(model_path, Ms, testdata, N_test=200):

    mu_matrix = np.zeros((len(model_path), N_test))
    sigma_matrix = np.zeros((len(model_path), N_test))
    mu_individual_list = []
    sigma_individual_list = []


    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        mu, sigma, mus, sigmas = var_inference(model, testloader)

        mu_matrix[i, :] = mu
        sigma_matrix[i, :] = sigma
        mu_individual_list.append(mus)
        sigma_individual_list.append(sigmas)
            
    return mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list

def get_bnn_predictions(bnn_path, testdata, N_test=200):
    model = torch.load(bnn_path)

    testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=bnn_collate_fn, pin_memory=True)

    predictions = []
    stds = []

    for x_test, y_test in testloader:
        x_test, y_test = x_test.float(), y_test.float()
        with torch.no_grad():
            mu, rho = model(x_test, sample=True)
            sigma = model.get_sigma(rho)
            predictions.append(mu.cpu().detach().numpy())
            stds.append(sigma.cpu().detach().numpy())

    return predictions, stds

def main(model_name, model_path, Ms):
    df_test = pd.read_csv("data/toydata/test_data.csv")
        
    x_test, y_test = np.array(list(df_test['x'])), np.array(list(df_test['y']))
    testdata = ToyDataset(x_test, y_test)

    match model_name:
        case "Baseline":
            pass
        case "MIMO":
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_var_mimo_predictions(model_path, Ms, testdata, N_test=200)
            np.savez(f'reports/Logs/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)
        case "Naive":
            mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = get_var_naive_predictions(model_path, Ms, testdata, N_test=200)
            np.savez(f'reports/Logs/{model_name}', predictions = mu_matrix, mu_individual = mu_individual_list, predicted_std = sigma_matrix, sigma_individual = sigma_individual_list)
        case "BNN":
            predictions, stds = get_bnn_predictions(model_path, testdata, N_test=200)
            np.savez(f'reports/Logs/{model_name}', predictions = predictions, predicted_std = stds)
        case "MIBMO":
            pass

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='MIMO', help='Model name [Baseline, MIMO, Naive, BNN, MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    args = parser.parse_args()

    base_path = f'models/{args.model_name}'
    model_path = [model for model in glob.glob(os.path.join(base_path,'*.pt'))]
    Ms = [int(M) for M in args.Ms.split(',')]


