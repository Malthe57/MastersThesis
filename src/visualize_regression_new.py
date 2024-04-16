import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression_data, reliability_diagram_regression
from data.OneD_dataset import ToyDataset, load_toydata, generate_data
from data.MultiD_dataset import load_multireg_data, generate_multidim_data
from utils.utils import get_training_min_max
from utils.metrics import compute_regression_statistics

def destandardise(min, max, y, is_sigma=False):
    '''
    Destandardise outputs from standardised model.
    Inputs:
    - min: the min value used for standardisation of data
    - max: the max value used for standardisation of data
    - y: the data to be destandardised
    
    Outputs:
    - y: destandardised data
    '''
    a = 2/(max-min)
    b = (-2*min)/(max-min) - 1

    if is_sigma:
        return y/a
    else:
        return (y-b)/a



def plot_regression(mu, sigma, y, model_name, dataset, Ms):
    '''
    Plot regression results
    Inputs:
    - mu: mean of predictions
    - sigma: standard deviation of predictions
    - y: true values
    - model_name: name of model
    - M: number of subnetworks
    '''
    N_test = len(y)
    if dataset == 'toydata':
        x_test, line = generate_data(N_test, lower=-0.5, upper=1.5, std=0.00)
        traindata, _, testdata, _, _ = load_toydata(normalise=False)
        x_train, y_train = traindata.x, traindata.y
        y_test = testdata.y
    elif dataset == 'multitoydata':
        x_test, line = generate_multidim_data(N_test, lower=-0.5, upper=1.5, std=0.00)
        traindata, _, testdata, _, _, _, _ = load_multireg_data(dataset, standardise=False)
        y_test = testdata.y

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # plot data
    ax.grid()
    # ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=4, zorder=1)
    ax.plot(x_test, line, '--', label='True function', color='red', zorder=2)
    ax.plot(x_test, y, '.', label='Test data', color='black', markersize=4, zorder=0)
    
    # plot predicitons with confidence intervals
    for i in range(len(Ms)):
        if not model_name == 'BNN':
            ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} Predictions with {Ms[i]} members', linewidth=2)
            ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] + 1.96*sigma[i], alpha=0.2, label=f'Confidence Interval with {Ms[i]} members')
        else:
            ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} Predictions', linewidth=2)
            ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] + 1.96*sigma[i], alpha=0.2, label=f'Confidence Interval')

    ax.legend()
    plt.show()


if __name__ == '__main__':

    dataset = 'crimedata'
    models = ['MIMO','Naive','BNN','MIMBO']
    Ms = [2]
    reps = 1

    if dataset == 'toydata':
        _, _, testdata, _, test_length = load_toydata(normalise=True)
        standardise_min = -1
        standardise_max = 1
    else:
        _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset)

    #De-standardise data:
    testdata.y = destandardise(standardise_min, standardise_max, testdata.y)    

    for model in models:
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        for i, M in enumerate(Ms):
            mu = mu_matrix[:,i,:]
            mu = destandardise(standardise_min, standardise_max, mu)
            sigma = sigma_matrix[:,i,:]
            sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)
            expected_mu = np.mean(mu, axis=0)
            expected_sigma = np.mean(sigma, axis=0)
            RMSE = np.sqrt(np.mean(np.power(testdata.y - mu, 2), axis=1))
            expected_RMSE = np.sqrt(np.mean(np.power(testdata.y - expected_mu,2),axis=0))
            mean_sigma = np.mean(expected_sigma, axis=0) 
            GaussianNLL = np.mean(0.5*(expected_sigma)+np.power(expected_mu-testdata.y,2)/expected_sigma)

            if dataset == 'toydata' or dataset == 'multitoydata':
                plot_regression(mu, sigma, testdata.y, model, dataset, Ms=Ms)


            if model == 'BNN':
                print(f'\n Best RMSE of {model} on {dataset}: {np.min(RMSE)}')
                print(f'\n Expected MSE of {model} on {dataset} with {reps} repetitions: ', expected_RMSE)
                print(f'\n Expected Standard deviation of {model} on {dataset} with {reps} repetitions', mean_sigma)

            else:
                print(f'\n Best RMSE of {model} on {dataset} with {M} subnetworks: {np.min(RMSE)}')
                print(f'\n Expected MSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions: ', expected_RMSE)
                print(f'\n Expected Standard deviation of {model} on {dataset} with {M} subnetworks and {reps} repetitions', mean_sigma)
            print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions', GaussianNLL)
            print('\n -----------------------')
            #reliability_diagram_regression(mu, testdata.y, sigma, M=M, model_name=model)


            


