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



def plot_regression(mu, sigma, y, model_name, dataset, Ms, mu_individual, sigma_individual):
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


    # compute epistemic and aleatoric uncertainty
    aleatoric = np.mean(np.power(sigma_individual, 2), axis=1)
    epistemic = np.mean(np.power(mu_individual,2), axis=1) - np.power(np.mean(mu_individual, axis=1), 2)

    aleatoric = np.sqrt(sigma**2 - epistemic) # the mixture variance, sigma**2, is the sum of the aleatoric and epistemic uncertainty

    # plot data
    ax.grid()
    # ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=4, zorder=1)
    ax.plot(x_test, line, '--', label='True function', color='red', zorder=2)
    ax.plot(x_test, y, '.', label='Test data', color='black', markersize=2, zorder=0)
    
    # plot predicitons with confidence intervals
    for i in range(len(Ms)):
        # compute aleatoric and epsitemic uncertainty
        pass


        if not model_name == 'BNN':
            ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} Predictions with {Ms[i]} members', linewidth=2)
            ax.fill_between(x_test, mu[i] - 1.96*aleatoric[i], mu[i] + 1.96*aleatoric[i], alpha=0.3, label=f'Aleatoric uncertainty with {Ms[i]} members')
            # ax.fill_between(x_test, mu[i] - 1.96*epistemic[i], mu[i] + 1.96*epistemic[i], alpha=0.3, label=f'Aleatoric + epistemic uncertainty with {Ms[i]} members')
            # plot aleatoric + epistemic uncertainty 'outside' the aleatoric uncertainty
            ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] - 1.96*aleatoric[i], alpha=0.5, color='orange', label=f'Aleatoric + epistemic uncertainty with {Ms[i]} members')
            ax.fill_between(x_test, mu[i] + 1.96*aleatoric[i], mu[i] + 1.96*sigma[i], alpha=0.5, color='orange')
        else:
            ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} Predictions', linewidth=2)
            ax.fill_between(x_test, mu[i] - 1.96*aleatoric[i], mu[i] + 1.96*aleatoric[i], alpha=0.3, label=f'Aleatoric uncertainty')
            # plot aleatoric + epistemic uncertainty 'outside' the aleatoric uncertainty
            ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] - 1.96*aleatoric[i], alpha=0.5, color='orange', label=f'Aleatoric + epistemic uncertainty ')
            ax.fill_between(x_test, mu[i] + 1.96*aleatoric[i], mu[i] + 1.96*sigma[i], alpha=0.5, color='orange')

    ax.legend()
    plt.show()

def calculate_statistics(mu, sigma, y):
    RMSE = np.sqrt(np.mean(np.power(y - mu, 2), axis=1)) # compute RMSE
    GaussianNLL = np.mean(0.5*(sigma)+np.power(mu-y,2)/sigma, axis=1) # compute GNNL
    best_idx = np.argmin(GaussianNLL)

    return RMSE, GaussianNLL, best_idx

if __name__ == '__main__':

    dataset = 'multitoydata'
    models = ['MIMO']
    Ms = [1,2,3,4,5]
    reps = 5
    best_idxs = []

    if dataset == 'toydata':
        _, _, testdata, _, test_length = load_toydata(normalise=True)
        standardise_min = -1
        standardise_max = 1
    else:
        _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset)

    #De-standardise data:
    y = destandardise(standardise_min, standardise_max, testdata.y)   

    #Get idx for out-of-distribution testdata:
    x_test = np.linspace(-0.5, 1.5, 5000)
    ood_idx = np.logical_or(x_test<-0.25, x_test>1.0)
    id_idx = np.logical_and(x_test >= -0.25, x_test <= 1.0)
    for model in models:
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        for i, M in enumerate(Ms):
            
            mu = mu_matrix[:,i,:]
            mu_individual = mu_individual_list[:,:, :sum(Ms[:i+1])] # get individual predictions for 0:1, 1:3, 3:6 etc in mu_individual_list
            mu = destandardise(standardise_min, standardise_max, mu)
            mu_individual = destandardise(standardise_min, standardise_max, mu_individual)
            sigma = sigma_matrix[:,i,:]
            sigma_individual = sigma_individual_list[:,:, :sum(Ms[:i+1])] # get individual standard deviations for 0:1, 1:3, 3:6 etc in sigma_individual_list
            sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)
            sigma_individual = destandardise(standardise_min, standardise_max, sigma_individual, is_sigma=True)

            # in-distribution metrics
            RMSE, GNLL, best_idx = calculate_statistics(mu[:, id_idx], sigma[:, id_idx], y[id_idx])
            best_idxs.append(best_idx)

            # out-of-distribution metrics
            RMSE_ood, GNLL_ood, best_idx_ood = calculate_statistics(mu[:, ood_idx], sigma[:, ood_idx], y[ood_idx])            

            if dataset == 'toydata' or dataset == 'multitoydata':
                plot_regression(mu[best_idx].reshape(1,-1), sigma[best_idx].reshape(1,-1), y, model, dataset, Ms = [M], mu_individual = mu_individual[best_idx], sigma_individual = sigma_individual[best_idx])
                None

                if model == 'BNN':
                    print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
                    print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
                    print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n In-distribution: {np.mean(RMSE)} ± {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} ± {1.96*np.std(RMSE_ood)/reps}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} ± {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} ± {1.96*np.std(GNLL_ood)/reps}')
                    # print(f'\n Expected Standard deviation of {model} on {dataset} with {reps} repetitions', np.mean(sigma))
                    
                else:
                    print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
                    print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
                    print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {np.mean(RMSE)} ± {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} ± {1.96*np.std(RMSE_ood)/reps}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} ± {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} ± {1.96*np.std(GNLL_ood)/reps}')
                    # print(f'\n Expected Standard deviation of {model} on {dataset} with {M} subnetworks and {reps} repetitions', np.mean(sigma))
                    
                reliability_diagram_regression(mu[:, id_idx], y[id_idx], sigma[:, id_idx], M=M, model_name=model+'_id')
                reliability_diagram_regression(mu[:, ood_idx], y[ood_idx], sigma[:, ood_idx], M=M, model_name=model+'_ood')
                reliability_diagram_regression(mu, y, sigma, M=M, model_name = model)
                print('\n -----------------------')

