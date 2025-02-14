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



def plot_regression(mu, sigma, y, model_name, dataset, Ms, mu_individual, sigma_individual, standardise_min, standardise_max, ood):
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
        traindata, _, testdata, _, _, _, _ = load_toydata(normalise=False)
        x_train, y_train = traindata.x, traindata.y
    elif dataset == 'multitoydata':
        x_test, line = generate_multidim_data(N_test, lower=-0.5, upper=1.5, std=0.00)
        traindata, _, testdata, _, _, _, _= load_multireg_data(dataset, standardise=True)
        x_train = np.load('data/multidimdata/toydata/x_1d.npz')['x_1d']
        y_train = traindata.y
        y_train = destandardise(standardise_min, standardise_max, traindata.y) 

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True) 

    # compute epistemic and aleatoric uncertainty
    aleatoric = np.mean(np.power(sigma_individual, 2), axis=1)
    epistemic = np.mean(np.power(mu_individual,2), axis=1) - np.power(np.mean(mu_individual, axis=1), 2)

    aleatoric_std = np.sqrt(sigma**2 - epistemic) # the mixture variance, sigma**2, is the sum of the aleatoric and epistemic uncertainty

    # plot data
    ax.grid()
    # ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=4, zorder=1)
    ax.plot(x_test, line, '--', label='True function', color='red', zorder=2)
    # ax.plot(x_test, y, '.', label='Test data', color='black', markersize=2, zorder=0)
    ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=2, zorder=0)
    
    # plot predicitons with confidence intervals
    # for i in range(len(Ms)):

    #     if not model_name == 'BNN':
    #         ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} M={Ms[i]} prediction', linewidth=2) if Ms[i] > 1 else ax.plot(x_test, mu[i], '-', label=f'Mean Baseline prediction', linewidth=2)
    #         ax.fill_between(x_test, mu[i] - 1.96*aleatoric_std[i], mu[i] + 1.96*aleatoric_std[i], alpha=0.3, label=f'Aleatoric uncertainty')
    #         # ax.fill_between(x_test, mu[i] - 1.96*epistemic[i], mu[i] + 1.96*epistemic[i], alpha=0.3, label=f'Aleatoric + epistemic uncertainty with {Ms[i]} members')
    #         # plot aleatoric + epistemic uncertainty 'outside' the aleatoric uncertainty
    #         if mu_individual.shape[1] > 1:
    #             ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] - 1.96*aleatoric_std[i], alpha=0.5, color='orange', label=f'Aleatoric + epistemic uncertainty')
    #             ax.fill_between(x_test, mu[i] + 1.96*aleatoric_std[i], mu[i] + 1.96*sigma[i], alpha=0.5, color='orange')
    #         for j in range(mu_individual.shape[1]):
    #             ax.plot(x_test, mu_individual[:,j], alpha=0.1, color='blue')

    #     else:
    #         ax.plot(x_test, mu[i], '-', label=f'Mean {model_name} prediction', linewidth=2)
    #         ax.fill_between(x_test, mu[i] - 1.96*aleatoric_std[i], mu[i] + 1.96*aleatoric_std[i], alpha=0.3, label=f'Aleatoric uncertainty ')
    #         # plot aleatoric + epistemic uncertainty 'outside' the aleatoric uncertainty
    #         ax.fill_between(x_test, mu[i] - 1.96*sigma[i], mu[i] - 1.96*aleatoric_std[i], alpha=0.5, color='orange', label=f'Aleatoric + epistemic uncertainty')
    #         ax.fill_between(x_test, mu[i] + 1.96*aleatoric_std[i], mu[i] + 1.96*sigma[i], alpha=0.5, color='orange')
    #         for j in range(mu_individual.shape[1]):
    #             ax.plot(x_test, mu_individual[:,j], alpha=0.1, color='blue')
        
    ax.legend()
    if ood:
        ax.set_xlim(-0.5, 1.5)
    else:
        ax.set_xlim(-0.25, 1.0)
        ax.set_ylim(-1.5,1.5)

    os.makedirs(f"reports/figures/plots/regression/{dataset}/", exist_ok=True)
    plt.savefig(f"reports/figures/plots/regression/{dataset}/{model_name}_M{Ms[0]}_{dataset}_regression.png", dpi=1200, bbox_inches='tight')

    plt.show()

def calculate_statistics(mu, sigma, y):
    RMSE = np.sqrt(np.mean(np.power(y - mu, 2), axis=1)) # compute RMSE
    GaussianNLL = np.mean(0.5*(np.log(np.power(sigma,2))+np.power(mu-y,2)/np.power(sigma,2)), axis=1) # compute GNNL
    # verified with:
    # import torch.nn as nn
    # loss = nn.GaussianNLLLoss()
    # losses = []
    # for i in range(5):
    #     losses.append(loss(torch.tensor(mu)[i], torch.tensor(y), torch.tensor(sigma).pow(2)[i]))
    
    # print(GaussianNLL)
    # best_idx = np.argin(GaussianNLL)
    best_idx = np.argmax(GaussianNLL)    
    print("Visualising argmax")
    # print(best_idx)

    return RMSE, GaussianNLL, best_idx

def visualise_toydata(dataset, models, Ms, ood, reps):
    best_idxs = []

    if dataset == 'toydata':
        _, _, testdata, _, test_length, standardise_max, standardise_min = load_toydata(normalise=True)
    else:
        _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset, standardise=True)

    #De-standardise data:
    y = destandardise(standardise_min, standardise_max, testdata.y) 
    # y = testdata.y  

    #Get idx for out-of-distribution testdata:
    if dataset == 'multitoydata':
        x_test = np.linspace(-0.5, 1.5, 5000)
        ood_idx = np.logical_or(x_test<-0.25, x_test>1.0)
        id_idx = np.logical_and(x_test >= -0.25, x_test <= 1.0)
    elif dataset == 'toydata':
        x_test = np.linspace(-0.5, 1.5, 500)
        ood_idx = np.logical_or(x_test<-0.25, x_test>1.0)
        id_idx = np.logical_and(x_test >= -0.25, x_test <= 1.0)
        
    for model in models:
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        for i, M in enumerate(Ms):
            
            mu = mu_matrix[:,i,:]
            mu = destandardise(standardise_min, standardise_max, mu)
            sigma = sigma_matrix[:,i,:]
            sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)

            if model == 'BNN' or model =='MIMBO':
                mu_individual = mu_individual_list[:,:, i*10:i*10+10] # get individual predictions for 0:1, 1:3, 3:6 etc in mu_individual_list
                sigma_individual = sigma_individual_list[:,:, i*10:i*10+10] # get individual standard deviations for 0:1, 1:3, 3:6 etc in sigma_individual_list

            else:
                mu_individual = mu_individual_list[:,:, sum(Ms[:i+1])-M:sum(Ms[:i+1])] # get individual predictions for 0:1, 1:3, 3:6 etc in mu_individual_list
                sigma_individual = sigma_individual_list[:,:, sum(Ms[:i+1])-M:sum(Ms[:i+1])] # get individual standard deviations for 0:1, 1:3, 3:6 etc in sigma_individual_list

            mu_individual = destandardise(standardise_min, standardise_max, mu_individual)
            sigma_individual = destandardise(standardise_min, standardise_max, sigma_individual, is_sigma=True)

           
            if dataset == 'toydata' or dataset == 'multitoydata':
                # in-distribution metrics
                RMSE, GNLL, best_idx = calculate_statistics(mu[:, id_idx], sigma[:, id_idx], y[id_idx])
                best_idxs.append(best_idx)

                # out-of-distribution metrics
                RMSE_ood, GNLL_ood, best_idx_ood = calculate_statistics(mu[:, ood_idx], sigma[:, ood_idx], y[ood_idx])            

                plot_regression(mu[best_idx].reshape(1,-1), sigma[best_idx].reshape(1,-1), y, model, dataset, Ms = [M], mu_individual = mu_individual[best_idx], sigma_individual = sigma_individual[best_idx], standardise_min=standardise_min, standardise_max=standardise_max, ood=ood)
                None

                if model == 'BNN':
                    # print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
                    # print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
                    print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} \pm {1.96*np.std(RMSE_ood)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} \pm {1.96*np.std(GNLL_ood)/np.sqrt(reps)}')
                    # print(f'\n Expected Standard deviation of {model} on {dataset} with {reps} repetitions', np.mean(sigma))
                    
                else:
                    # print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
                    # print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
                    print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} \pm {1.96*np.std(RMSE_ood)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} \pm {1.96*np.std(GNLL_ood)/np.sqrt(reps)}')
                    # print(f'\n Expected Standard deviation of {model} on {dataset} with {M} subnetworks and {reps} repetitions', np.mean(sigma))
                    
                reliability_diagram_regression(mu[:, id_idx], y[id_idx], sigma[:, id_idx], M=M, model_name=model, dataset=dataset, ood=False)
                reliability_diagram_regression(mu[:, ood_idx], y[ood_idx], sigma[:, ood_idx], M=M, model_name=model, dataset=dataset, ood=True)
                # reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset)
                print('\n -----------------------')

def visualise_crimedata(dataset, models, Ms, ood, reps):
    predicted_variances = []
    best_idxs = []

    _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset, standardise=True, ood=ood)

    #De-standardise data:
    y = destandardise(standardise_min, standardise_max, testdata.y) 
    # y = testdata.y  
        
    for model in models:
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}_id.npz") if ood == False else np.load(f"reports/Logs/{model}/{dataset}/{model}_ood.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        for i, M in enumerate(Ms):
            
            mu = mu_matrix[:,i,:]
            mu = destandardise(standardise_min, standardise_max, mu)
            sigma = sigma_matrix[:,i,:]
            sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)

            RMSE, GNLL, best_idx = calculate_statistics(mu, sigma, y)
            best_idxs.append(best_idx)

            if ood:
                if model == 'BNN':
                    print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n Out-of-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n Out-of-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
                else:
                    print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n Out-of-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n Out-of-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
                reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset, ood=ood)
                print('\n -----------------------')
            else:

                if model == 'BNN':
                    print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
                else:
                    print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
                    print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
                reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset, ood=ood)
                print('\n -----------------------')

def visualise_variances(ood=False):
    dataset = 'crimedata'
    models = ['MIMO', 'Naive', 'BNN', 'MIMBO']
    Ms_list = [[1,2,3,4,5], [2,3,4,5], [1], [2,3,4,5]]
    predicted_variances_list = []
    labels = []

    _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset, standardise=True, ood=ood)
    y = destandardise(standardise_min, standardise_max, testdata.y) 

    for model, Ms in zip(models, Ms_list):
        Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}_id.npz") if ood == False else np.load(f"reports/Logs/{model}/{dataset}/{model}_ood.npz")
        mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
        
        predicted_variances = np.power(sigma_matrix, 2)
        for i, M in enumerate(Ms):
            mu = mu_matrix[:,i,:]
            mu = destandardise(standardise_min, standardise_max, mu)
            sigma = sigma_matrix[:,i,:]
            sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)

            RMSE, GNLL, best_idx = calculate_statistics(mu, sigma, y)
            predicted_variances_list.append(predicted_variances[best_idx, i, :])
            if M == 1:
                if model == 'MIMO':
                    labels.append(f'Baseline')
                elif model == 'BNN':
                    labels.append(f'{model}')
            else:
                labels.append(f'{model} M={M}')

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True)
    ax.violinplot(np.array(predicted_variances_list).T, positions=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    ax.set_xticklabels(labels)
    plt.legend()
    plt.show()
    



if __name__ == '__main__':
    dataset = 'toydata'
    models = ['BNN']
    Ms = [1]
    ood = True
    reps = 5
    # visualise_variances()
    if dataset == 'crimedata':
        visualise_crimedata(dataset, models, Ms, ood, reps)
    else:
        visualise_toydata(dataset, models, Ms, ood, reps)

    
    # dataset = 'toydata'
    # models = ['MIMBO']
    # Ms = [2,3,4,5]
    # ood = False
    # reps = 5
    # best_idxs = []

    # if dataset == 'toydata':
    #     _, _, testdata, _, test_length = load_toydata(normalise=True)
    #     standardise_min = -1
    #     standardise_max = 1
    # else:
    #     _, _, testdata, _, test_length, standardise_max, standardise_min = load_multireg_data(dataset, standardise=True, ood=ood)

    # #De-standardise data:
    # y = destandardise(standardise_min, standardise_max, testdata.y) 
    # # y = testdata.y  

    # #Get idx for out-of-distribution testdata:
    # if dataset == 'multitoydata':
    #     x_test = np.linspace(-0.5, 1.5, 5000)
    #     ood_idx = np.logical_or(x_test<-0.25, x_test>1.0)
    #     id_idx = np.logical_and(x_test >= -0.25, x_test <= 1.0)
    # elif dataset == 'toydata':
    #     x_test = np.linspace(-0.5, 1.5, 500)
    #     ood_idx = np.logical_or(x_test<-0.25, x_test>1.0)
    #     id_idx = np.logical_and(x_test >= -0.25, x_test <= 1.0)
        
    # for model in models:
    #     Results =  np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
    #     mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list = Results['predictions'], Results['predicted_std'], Results['mu_individual'], Results['sigma_individual']
    #     for i, M in enumerate(Ms):
            
    #         mu = mu_matrix[:,i,:]
    #         mu = destandardise(standardise_min, standardise_max, mu)
    #         sigma = sigma_matrix[:,i,:]
    #         sigma = destandardise(standardise_min, standardise_max, sigma, is_sigma=True)

    #         if model == 'BNN' or model =='MIMBO':
    #             mu_individual = mu_individual_list[:,:, i*10:i*10+10] # get individual predictions for 0:1, 1:3, 3:6 etc in mu_individual_list
    #             sigma_individual = sigma_individual_list[:,:, i*10:i*10+10] # get individual standard deviations for 0:1, 1:3, 3:6 etc in sigma_individual_list

    #         else:
    #             mu_individual = mu_individual_list[:,:, sum(Ms[:i+1])-M:sum(Ms[:i+1])] # get individual predictions for 0:1, 1:3, 3:6 etc in mu_individual_list
    #             sigma_individual = sigma_individual_list[:,:, sum(Ms[:i+1])-M:sum(Ms[:i+1])] # get individual standard deviations for 0:1, 1:3, 3:6 etc in sigma_individual_list

    #         mu_individual = destandardise(standardise_min, standardise_max, mu_individual)
    #         sigma_individual = destandardise(standardise_min, standardise_max, sigma_individual, is_sigma=True)

           
    #         if dataset == 'toydata' or dataset == 'multitoydata':
    #             # in-distribution metrics
    #             RMSE, GNLL, best_idx = calculate_statistics(mu[:, id_idx], sigma[:, id_idx], y[id_idx])
    #             best_idxs.append(best_idx)

    #             # out-of-distribution metrics
    #             RMSE_ood, GNLL_ood, best_idx_ood = calculate_statistics(mu[:, ood_idx], sigma[:, ood_idx], y[ood_idx])            

    #             plot_regression(mu[best_idx].reshape(1,-1), sigma[best_idx].reshape(1,-1), y, model, dataset, Ms = [M], mu_individual = mu_individual[best_idx], sigma_individual = sigma_individual[best_idx], standardise_min=standardise_min, standardise_max=standardise_max)
    #             None

    #             if model == 'BNN':
    #                 # print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
    #                 # print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
    #                 print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} \pm {1.96*np.std(RMSE_ood)/np.sqrt(reps)}')
    #                 print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} \pm {1.96*np.std(GNLL_ood)/np.sqrt(reps)}')
    #                 # print(f'\n Expected Standard deviation of {model} on {dataset} with {reps} repetitions', np.mean(sigma))
                    
    #             else:
    #                 # print(f'\n Best RMSE of {model} on {dataset}:\n In-distribution: {np.min(RMSE)} \n Out-of-distribution: {np.min(RMSE_ood)}')
    #                 # print(f'\n best Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {GNLL[best_idx]}\n Out-of-distribution: {GNLL_ood[best_idx]}')
    #                 print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(RMSE_ood)} \pm {1.96*np.std(RMSE_ood)/np.sqrt(reps)}')
    #                 print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)} \n Out-of-distribution: {np.mean(GNLL_ood)} \pm {1.96*np.std(GNLL_ood)/np.sqrt(reps)}')
    #                 # print(f'\n Expected Standard deviation of {model} on {dataset} with {M} subnetworks and {reps} repetitions', np.mean(sigma))
                    
    #             reliability_diagram_regression(mu[:, id_idx], y[id_idx], sigma[:, id_idx], M=M, model_name=model, dataset=dataset, ood=False)
    #             reliability_diagram_regression(mu[:, ood_idx], y[ood_idx], sigma[:, ood_idx], M=M, model_name=model, dataset=dataset, ood=True)
    #             # reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset)
    #             print('\n -----------------------')

    #         else:
    #             RMSE, GNLL, best_idx = calculate_statistics(mu, sigma, y)
    #             best_idxs.append(best_idx)

    #             if ood:
    #                 if model == 'BNN':
    #                     print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n Out-of-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
    #                     print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n Out-of-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
    #                 else:
    #                     print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n Out-of-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
    #                     print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n Out-of-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
    #                 reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset)
    #                 print('\n -----------------------')
    #             else:

    #                 if model == 'BNN':
    #                     print(f'\n Expected RMSE of {model} on {dataset} with {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
    #                     print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
    #                 else:
    #                     print(f'\n Expected RMSE of {model} on {dataset} with {M} subnetworks and {reps} repetitions:\n In-distribution: {np.mean(RMSE)} \pm {1.96*np.std(RMSE)/np.sqrt(reps)}')
    #                     print(f'\n Expected Gaussian NLL on test data of {model} on {dataset} with {M} subnetworks and {reps} repetitions: \n In-distribution:  {np.mean(GNLL)} \pm {1.96*np.std(GNLL)/np.sqrt(reps)}')
    #                 reliability_diagram_regression(mu, y, sigma, M=M, model_name = model, dataset=dataset, ood=ood)
    #                 print('\n -----------------------')
