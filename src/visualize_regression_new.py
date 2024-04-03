import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, plot_regression_data, reliability_diagram_regression
from data.OneD_dataset import ToyDataset, load_toydata
from data.MultiD_dataset import load_multireg_data
from utils.utils import get_training_min_max
from utils.metrics import compute_regression_statistics



if __name__ == '__main__':

    dataset = 'newsdata'
    models = ['MIMBO']
    Ms = [2]
    reps = 2

    if dataset == 'toydata':
        _, _, testdata, _, test_length = load_toydata(normalise=True)
    else:
        _, _, testdata, _, test_length = load_multireg_data(dataset)

    

    for model in models:
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        for i, M in enumerate(Ms):
            mu = mu_matrix[:,i,:]
            sigma = sigma_matrix[:,i,:]
            expected_mu = np.mean(mu, axis=0)
            expected_sigma = np.sqrt(np.mean((np.power(mu,2) + np.power(sigma,2)), axis=0) - np.power(expected_mu,2))
            expected_RMSE = np.sqrt(np.mean(np.power(testdata.y - expected_mu,2),axis=0))
            mean_sigma = np.mean(expected_sigma, axis=0)

            if model == 'BNN':
                print(f'\n Expected MSE of {model} on {dataset} with {reps} repetitions: ', expected_RMSE)
                print(f' Expected Standard deviation of {model} on {dataset} with {reps} repetitions', mean_sigma)

            else:
                print(f'\n Expected MSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions: ', expected_RMSE)
                print(f'\n Expected Standard deviation of {model} on {dataset} with {M} subnetworks and {reps} repetitions', mean_sigma)

            reliability_diagram_regression(expected_mu, testdata.y, expected_sigma, M=M, model_name=model)


            
            

