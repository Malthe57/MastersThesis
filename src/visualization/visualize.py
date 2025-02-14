import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import StepPatch
import torch
from torchvision.transforms import transforms
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from visualization.PCA import PCA
from utils.utils import make_dirs   
from data.CIFAR10 import load_cifar10, load_CIFAR10C
from data.CIFAR100 import load_cifar100, load_CIFAR100C

def plot_loss(losses, val_losses, model_name="MIMO", task='regression'):

    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle(f"{model_name} Losses")

    ax[0].plot(losses, label='Train loss')
    ax[0].set_title('Train loss')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].grid()

    ax[1].plot(val_losses, label='Validation loss', color='orange')
    ax[1].set_title('Validation loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')
    ax[1].grid()
    plt.savefig(f"reports/figures/losses/{task}/{model_name}_losses.png")  
    plt.show()


def plot_log_probs(log_priors, log_variational_posteriors, NLLs, model_name="BNN", task='regression'):

    fig, ax = plt.subplots(1,3, figsize=(18,6))

    ax[0].plot(log_priors, label='Train log prior')
    ax[0].set_title('Train log prior')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Log prior')
    ax[0].grid()

    ax[1].plot(log_variational_posteriors, label='Train log variational posterior', color='orange')
    ax[1].set_title('Train log variational posterior')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Log variational posterior')
    ax[1].grid()

    ax[2].plot(NLLs, label='Train NLL', color='green')
    ax[2].set_title('Train NLL')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('NLL')
    ax[2].grid()
    plt.savefig(f"reports/figures/losses/{task}/{model_name}_log_probs.png")
    plt.show()

def plot_weight_distribution(MIMO_model, Naive_model, mode = 'Classification'):
# weight distribution
    weights_mimo = []
    for param in MIMO_model.parameters():
        weights_mimo.extend(param.flatten().cpu().detach().numpy())

    weights_naive = []
    for param in Naive_model.parameters():
        weights_naive.extend(param.flatten().cpu().detach().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(weights_mimo, bins=50, alpha=0.5, label='MIMO_model')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Weight Distribution - Classification MIMO Model')

    ax2.hist(weights_naive, bins=50, alpha=0.5, label='naive_model')
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Weight Distribution - Classification Naive Model')

    plt.tight_layout()
    plt.savefig(f"reports/figures/{mode}_model_weights")
    plt.show()

def plot_regression(x_train, y_train, x_test, y_test, line, mu_pred_matrix, stds, Ms, model_name='MIMO', save_fig=True):
    # plot data
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    ### plot data ###
    ax.grid()
    ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=4)
    ax.plot(x_test, line, '--', label='True function', color='red')
    # plot test data
    ax.plot(x_test, y_test, '.', label='Test data', color='black', markersize=4)

    # plot predicitons with confidence intervals
    for i in range(len(Ms)):
        if not model_name == 'BNN':
            ax.plot(x_test, mu_pred_matrix[i], '-', label=f'Mean {model_name} Predictions with {Ms[i]} members', linewidth=2)
            ax.fill_between(x_test, mu_pred_matrix[i] - 1.96*stds[i], mu_pred_matrix[i] + 1.96*stds[i], alpha=0.2, label=f'Confidence Interval with {Ms[i]} members')
        else:
            ax.plot(x_test, mu_pred_matrix[i], '-', label=f'Mean {model_name} Predictions', linewidth=2)
            ax.fill_between(x_test, mu_pred_matrix[i] - 1.96*stds[i], mu_pred_matrix[i] + 1.96*stds[i], alpha=0.2, label=f'Confidence Interval')

    ax.legend()

    if save_fig:
        plt.savefig(f"reports/figures/plots/regression/{model_name}_regression.png")

    plt.show()

def plot_regression_data(x_train, y_train, x_test, y_test, line, save_fig=True):

    fig, ax = plt.subplots(1,1, figsize=(12,6))
    ax.set_title("One-dimensional regression data")
    ### plot data ###
    ax.grid()
    ax.plot(x_train, y_train, '.', label='Train data', color='orange', markersize=4)
    ax.plot(x_test, line, '--', label='True function', color='red')
    # plot test data
    ax.plot(x_test, y_test, '.', label='Test data', color='black', markersize=4)

    ax.legend()

    if save_fig:
        plt.savefig(f"reports/figures/plots/regression/data_plot.png")
    plt.show()

def reliability_plot_classification(correct_predictions, confidence, naive_correct_predictions, naive_confidence, model_name, naive_model_name, M):
        #Code for generating reliability diagram:
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8,4))

    linspace = np.arange(0, 1.1, 0.1)
    bins_range = np.quantile(confidence, linspace)
    n_samples = len(correct_predictions)

    conf_step_height = np.zeros(10)
    acc_step_height = np.zeros(10)
    ECE_values = np.zeros(10)
    for i in range(len(bins_range)-1):
        loc = np.where(np.logical_and(confidence>=bins_range[i], confidence<bins_range[i+1]))
        conf_step_height[i] = np.mean(confidence[loc])
        acc_step_height[i] = np.mean(correct_predictions[loc])
        if np.isnan(conf_step_height[i]) == False:
            ECE_values[i] = len(loc[0])/n_samples*np.abs(acc_step_height[i]-conf_step_height[i])
        else:
            acc_step_height[i] = 0.0
    
    ECE = np.sum(ECE_values)/n_samples
    print(f"MIMO M{M} ECE: {ECE}")

    naive_conf_step_height = np.zeros(10)
    naive_acc_step_height = np.zeros(10)
    naive_ECE_values = np.zeros(10)
    for i in range(len(bins_range)-1):
        loc = np.where(np.logical_and(naive_confidence>=bins_range[i], naive_confidence<bins_range[i+1]))
        naive_conf_step_height[i] = np.mean(naive_confidence[loc])
        naive_acc_step_height[i] = np.mean(naive_correct_predictions[loc])
        if np.isnan(naive_conf_step_height[i]) == False:
            
            naive_ECE_values[i] = len(loc[0])/n_samples*np.abs(naive_acc_step_height[i]-naive_conf_step_height[i])
        else:
            naive_acc_step_height[i] = 0.0   

    naive_ECE = np.sum(naive_ECE_values)
    print(f"Naive M{M} ECE: {naive_ECE}")
    
    fig.supxlabel("Confidence")
    fig.supylabel("Accuracy")
    fig.suptitle(f'Reliability Diagrams for M={M}')
    fig.set_layout_engine('compressed')
    
    ax[0].grid(linestyle='dotted', zorder=0)
    ax[0].stairs(acc_step_height, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    ax[0].stairs(conf_step_height, bins_range, baseline = acc_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
    ax[0].plot(bins_range, bins_range, linestyle='--', color='gray', zorder=3)
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("MIMO")
    ax[0].legend()
    ax[0].text(0.7, 0.05, f'ECE={np.round(ECE,5)}', backgroundcolor='lavender', alpha=1.0, fontsize=8.0)

    ax[1].grid(linestyle='dotted')
    ax[1].stairs(naive_acc_step_height, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs')
    ax[1].stairs(naive_conf_step_height, bins_range, baseline = naive_acc_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap')
    ax[1].plot(bins_range, bins_range, linestyle='--', color='gray', )
    ax[1].text(0.7, 0.05, f'ECE={np.round(naive_ECE,5)}', backgroundcolor='lavender', alpha=1.0, fontsize=8.0)
    
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Naive Multiheaded")
    ax[1].legend()
    # plt.tight_layout()
    plt.savefig(f"reports/figures/{model_name}_confidence_plots.png")
    plt.show()

def reliability_plot_classification_single(correct_predictions, confidence, model_name, dataset, M=1, severity=5):
        #Code for generating reliability diagram:
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6,6), tight_layout=True)

    reps = correct_predictions.shape[0]
    linspace = np.arange(0, 1.1, 0.1)
    # bins_range = np.quantile(confidence.flatten(), linspace)
    bins_range = linspace
    n_samples = len(correct_predictions.T)
    
    conf_step_height = np.zeros((reps, 10))
    acc_step_height = np.zeros((reps,10))

    lengths = np.zeros((reps, 10))
    ECEs = np.zeros((reps, 10))
    for j in range(reps):
        for i in range(10):
            loc = np.where(np.logical_and(confidence[j,:]>bins_range[i], confidence[j,:]<=bins_range[i+1]))[0]
            if correct_predictions[j,loc].shape[0] != 0:
                acc_step_height[j, i] = np.mean(correct_predictions[j, loc])
                conf_step_height[j, i] = np.mean(confidence[j, loc])
                lengths[j, i] = correct_predictions[j,loc].shape[0]
                ECEs[j,i] = np.abs(acc_step_height[j, i]-conf_step_height[j, i])*lengths[j,i]
    
    ECE = np.sum(ECEs, axis=1)/n_samples
    # MSE_step_std = MSE_step_height[MSE_step_height!=0].std(axis=0)
    acc_step_std = np.zeros(10)
    acc_final_step = np.zeros(10)
    conf_final_step = np.zeros(10)
    
    for j, values in enumerate(acc_step_height.T):
        if np.all(np.array(values) == 0):
            acc_step_std[j] = 0
            acc_final_step[j] = 0
        else:
            acc_step_std[j] = np.std(values[values!=0])
            acc_final_step[j] = np.mean(values[values!=0])
    
    for j, values in enumerate(conf_step_height.T):
        if np.all(np.array(values) == 0):
            conf_final_step[j] = 0
        else:
            conf_final_step[j] = np.mean(values[values!=0])

    

    acc_step_ub = acc_final_step + 1.96*acc_step_std
    acc_step_lb = acc_final_step - 1.96*acc_step_std
    acc_sterr =  1.96*acc_step_std/np.sqrt(reps)
    bins_width = bins_range[1:]-bins_range[:-1]
    
    # fig.supxlabel("Confidence")
    # fig.supylabel("Accuracy")
    ax.set_xlabel("Confidence", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    # fig.set_layout_engine('compressed')

    min = lengths.sum(0).min()
    max = lengths.sum(0).max()
    norm = Normalize(vmin=0, vmax=1)

    # Create a colormap object
    cmap = plt.get_cmap('Blues')        

    # Create a ScalarMappable to map normalized values to colormap
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # define colors
    colors = sm.to_rgba(lengths.sum(0) / lengths.sum())

    cb = plt.colorbar(sm, ax=ax, fraction=0.042, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label='Sample density', fontsize=16)
    
    ax.grid(linestyle='dotted', zorder=0)
    # ax.stairs(acc_final_step, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    # ax.stairs(acc_step_ub, bins_range, baseline = acc_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='CI upper bound', zorder=2)
    # ax.stairs(acc_step_lb, bins_range, baseline = acc_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label= 'CI lower bound', zorder=2)
    ax.bar(x=bins_range[:-1], height=acc_final_step, width=bins_width, align='edge', linewidth=1.0, edgecolor='black',zorder=1, color=colors, label='Outputs')
    ax.bar(x=bins_range[:-1], height=acc_final_step, width=bins_width, align='edge', linewidth=1.0, edgecolor='black',zorder=3, color=None, fill=False)
    ax.bar(x=bins_range[:-1], height=conf_final_step-acc_final_step, width=bins_width, align='edge', zorder=2, fill=False, edgecolor='red', color='r', hatch='/', bottom=acc_final_step, label='Deficit')
    ax.errorbar(x=bins_range[:-1]+(bins_range[1:]-bins_range[:-1])*0.5, y=acc_final_step, yerr=acc_sterr, capsize=3, zorder=5, fmt='none', color='black', label='95% CI')
    # ax.stairs(conf_step_height, bins_range, baseline = acc_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
    ax.plot(linspace, linspace, linestyle='--', color='gray', zorder=4)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', fontsize=14)
    # ax.text(0.6, 0.05, f'ECE={np.round(np.mean(ECE),4)} ± {np.round(1.96*np.std(ECE)/np.sqrt(reps),4)}', backgroundcolor='lavender', alpha=1.0, fontsize=20.0)

    # QT backend
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    plt.tight_layout()
    
    dist_name = ['in-distribution', 'out-of-distribution']
    arch_name = 'WideResnet' if model_name[-4:] == 'Wide' else 'MediumCNN'
    model_name = model_name[2:-4] if model_name[-4:] == 'Wide' else model_name[2:]
    dataset = 'CIFAR10C' if dataset=='CIFAR10_C' else dataset
    # create directory 
    if severity is not None:
        in_dist = 1
        os.makedirs(f"reports/figures/reliability_diagrams/classification/{dataset}/{severity}", exist_ok=True)
    else:
        in_dist = 0
        os.makedirs(f"reports/figures/reliability_diagrams/classification/{dataset}", exist_ok=True)
    if M>1:
        ax.set_title(f"{arch_name}\n {model_name} M={M} on {dist_name[in_dist]} {dataset}", fontsize=16)
        print(f'ECE for {model_name} with {M} members: {np.mean(ECE)} ± {1.96*np.std(ECE)/np.sqrt(reps)}') 
        plt.savefig(f"reports/figures/reliability_diagrams/classification/{dataset}/{severity}/{model_name}_M{M}_{arch_name}_reliability_diagram.png", bbox_inches='tight', dpi=1200) if severity is not None else plt.savefig(f"reports/figures/reliability_diagrams/classification/{dataset}/{model_name}_M{M}_{arch_name}_reliability_diagram.png", bbox_inches='tight', dpi=1200)
    else:
        ax.set_title(f"{arch_name}\n {model_name} on {dist_name[in_dist]} {dataset}", fontsize=16)
        print(f'ECE for {model_name}: {np.mean(ECE)} ± {1.96*np.std(ECE)/np.sqrt(reps)}')  
        plt.savefig(f"reports/figures/reliability_diagrams/classification/{dataset}/{severity}/{model_name}_{arch_name}_reliability_diagram.png", bbox_inches='tight') if severity is not None else plt.savefig(f"reports/figures/reliability_diagrams/classification/{dataset}/{model_name}_{arch_name}_reliability_diagram.png", bbox_inches='tight')
    plt.show()

def get_ood_name(ood):
    if ood is None:
        return ' '
    elif ood == True:
        return ' out-of-distribution '
    elif ood == False:
        return ' in-distribution '

def reliability_diagram_regression(predictions, targets, predicted_std, M, dataset, model_name, ood=None):
    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout=True)
    
    reps = predictions.shape[0]
    predicted_variance = (predicted_std**2)
    # make bins from 
    linspace = np.arange(0, 1.1, 0.1)
    bins_range = np.quantile(predicted_variance.flatten(), linspace)
    n_samples = len(predictions.T)

    MSE_step_height = np.zeros((reps, 10))
    Variance_step_height = np.zeros((reps,10))

    squared_error = np.power(predictions - targets, 2)
    lengths = np.zeros((reps, 10))
    ECEs = np.zeros((reps, 10))
    for j in range(reps):
        for i in range(10):
            loc = np.where(np.logical_and(predicted_variance[j,:]>=bins_range[i], predicted_variance[j,:]<bins_range[i+1]))[0]
            if squared_error[j,loc].shape[0] != 0:
                MSE_step_height[j, i] = np.mean(squared_error[j, loc])
                Variance_step_height[j, i] = np.mean(predicted_variance[j, loc])
                lengths[j, i] = squared_error[j,loc].shape[0]
                ECEs[j,i] = np.abs(MSE_step_height[j, i]-Variance_step_height[j, i])*lengths[j,i]
    
    ECE = np.sum(ECEs, axis=1)/n_samples
    # MSE_step_std = MSE_step_height[MSE_step_height!=0].std(axis=0)
    MSE_step_std = np.zeros(10)
    MSE_final_step = np.zeros(10)
    
    for j, values in enumerate(MSE_step_height.T):
        if np.all(np.array(values) == 0):
            MSE_step_std[j] = 0
            MSE_final_step[j] = 0
        else:
            MSE_step_std[j] = np.std(values[values!=0])
            MSE_final_step[j] = np.mean(values[values!=0])
            
    Variance_step_height[Variance_step_height == 0] = np.nan
    mean_variance_step_height = np.nanmean(Variance_step_height,axis=0)

    MSE_sterr =  1.96*MSE_step_std/np.sqrt(reps)
    bins_width = bins_range[1:]-bins_range[:-1]

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linestyle='dotted', zorder=0)
    ax.bar(x=bins_range[:-1], height=MSE_final_step, width=bins_width, align='edge', linewidth=1.0, edgecolor='black',zorder=1, color='lightskyblue', label='Outputs')
    ax.bar(x=bins_range[:-1], height=MSE_final_step, width=bins_width, align='edge', linewidth=1.0, edgecolor='black',zorder=3, color=None, fill=False)
    ax.bar(x=bins_range[:-1], height=mean_variance_step_height-MSE_final_step, width=bins_width, align='edge', zorder=2, fill=False, edgecolor='red', color='r', hatch='/', bottom=MSE_final_step, label='Deficit to ideal calibration')
    ax.errorbar(x=np.sqrt(bins_range[:-1]*bins_range[1:]), y=MSE_final_step, yerr=MSE_sterr, capsize=3, zorder=5, fmt='none', color='black', label='95% CI')
    # plt.xlim(left=bins_range[0], right=bins_range[-1])
    ax.plot(bins_range, bins_range, linestyle='--', color='gray', zorder=4)
    plt.legend(fontsize=14)

    # ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    # plt.axis('equal')
    # ax.text(bins_range[0], MSE_final_step[-1]-0.2*MSE_final_step[-1], f'ECE={np.round(np.mean(ECE),4)} ± {np.round(1.96*np.std(ECE)/np.sqrt(reps),4)}', backgroundcolor='lavender', alpha=1.0, fontsize=10.0)


    plt.xlabel("Predicted variance", fontsize=20) 
    plt.ylabel("Mean squared error", fontsize=20) 
    # plt.axis('equal')
    # plt.xlim(left=bins_range[0], right=bins_range[-1])
    # plt.ylim(bottom=np.min(MSE_final_step), top=np.max(MSE_final_step))

    os.makedirs(f"reports/figures/reliability_diagrams/regression/{dataset}", exist_ok=True)
    if M > 1:
        ood_name = get_ood_name(ood)
        plt.title(f"{model_name} M={M} on{ood_name}{dataset}", fontsize=18)
        plt.savefig(f"reports/figures/reliability_diagrams/regression/{dataset}/{model_name}_{ood_name}_{M}_reliability_diagram.png", bbox_inches='tight')  
        print(f'ECE for {model_name} with {M} members: {np.mean(ECE)} \pm {1.96*np.std(ECE)/np.sqrt(reps)}') 
    else:
        ood_name = get_ood_name(ood)
        plt.title(f"{model_name} on{ood_name}{dataset}", fontsize=18) if model_name == 'BNN' else plt.title(f"Baseline on{ood_name}{dataset}", fontsize=18)
        plt.savefig(f"reports/figures/reliability_diagrams/regression/{dataset}/{model_name}_{ood_name}_reliability_diagram.png", bbox_inches='tight')
        print(f'ECE for {model_name}: {np.mean(ECE)} \pm {1.96*np.std(ECE)/np.sqrt(reps)}')  

    plt.show()


    
def function_space_plots(checkpoints, model_name, n_samples=20):
    '''
    Inputs:
    - checkpoints: a loaded list/array of validation predictions made at different point during training. 
    Should be in the shape [n_checkpoints, n_subnetworks, n_samples, n_classes] where n_samples is the number of data points used for prediction.
    - model_name: Name used for plot title
    - n_samples: number of predicted data points to use for plotting. Using too many could be troublesome.
    
    Outputs:
    - None, the function creates a tSNE plot of the checkpoints of different subnetworks.
    '''
    n_checkpoints, n_subnetworks, max_samples, _ = checkpoints.shape
    if n_samples > max_samples:
        print(f'the n_samples parameter is too large, reducing to the max value of {max_samples}')
        n_samples = max_samples

    checkpoint_list = checkpoints[:,:,:n_samples,:].numpy().reshape((-1,n_samples,10),order='F').reshape((n_samples*n_subnetworks,-1))
    tSNE = TSNE(n_components=2, perplexity=8.0, n_iter=2000)
    val_checkpoint_list2d = tSNE.fit_transform(checkpoint_list)
    color_options = ['r','g','b','y','c']
    colors = np.array([[color]*n_checkpoints for color in color_options[:n_subnetworks]]).flatten()
    plt.scatter(val_checkpoint_list2d[:,0], val_checkpoint_list2d[:,1], c=colors)
    for i in range(n_subnetworks):
        plt.plot(val_checkpoint_list2d[i*n_checkpoints:(i+1)*n_checkpoints,0], val_checkpoint_list2d[i*n_checkpoints:(i+1)*n_checkpoints,1], c=color_options[i], label=f'subnetwork {i}')
    plt.legend()
    plt.grid()
    plt.title(f't-SNE plot of subnetwork predictions for {model_name}')
    plt.show()


def pca(X, n_components):
    """
    Perform PCA on the dataset X and return the top n_components principal components.

    Parameters:
    X (numpy.ndarray): The input data matrix (n_samples, n_features).
    n_components (int): The number of principal components to return.

    Returns:
    X_pca (numpy.ndarray): The transformed data matrix with shape (n_samples, n_components).
    components (numpy.ndarray): The principal components with shape (n_components, n_features).
    explained_variance (numpy.ndarray): The amount of variance explained by each of the selected components.
    """
    # Step 1: Mean center the data
    X_meaned = X - np.mean(X, axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    # Step 5: Select the top n_components eigenvectors (principal components)
    components = sorted_eigenvectors[:, :n_components]

    # Step 6: Transform the data to the new space
    X_pca = np.dot(X_meaned, components)

    return X_pca


def multi_function_space_plots(checkpoints_list, model_names, dataset, architecture, n_samples=20, perplexity=10, num_components=2, use_axes=[1,2], algorithm='TSNE', twoD=True):
    '''
    Inputs:
    - Checkpoints_list: a list containing loaded checkpoint. It is assumed that the models listed have the same number of checkpoints, subnetworks and predicted classes
    - model_names: list of names used for plot title
    - n_samples: number of predictions used at each checkpoint. Caps out at the number of predictions saved in the checkpoint while training. Maximum n_samples should be the same for all models
    '''
    assert max(use_axes) <= num_components, 'The number of components to use for plotting is larger than the number of PCA components in the data'

    _ , _, n_classes, n_subnetworks, = checkpoints_list[0].shape
    n_checkpoints = [checkpoints.shape[0] for checkpoints in checkpoints_list]
    max_samples = min([checkpoints.shape[1] for checkpoints in checkpoints_list])
    if n_samples > max_samples:
        print(f'the n_samples parameter is too large, reducing to the max value of {max_samples}')
        n_samples = max_samples

    # check samples are the same for all models
    min_samples = min([c.shape[1] for c in checkpoints_list])

    # reshape checkpoints
    all_checkpoints = []
    for i, checkpoint in enumerate(checkpoints_list):
        # if i == 1:
        #     pass
        # else:
        reshaped_checkpoint = checkpoint[:, :n_samples, :, :].permute(0,3,1,2).numpy().reshape((-1, n_samples, n_classes), order='F').reshape((n_checkpoints[i]*n_subnetworks, -1), order='F')
        all_checkpoints.append(reshaped_checkpoint)

    # then concatenate
    all_checkpoints = np.concatenate(all_checkpoints, axis=0)


    if algorithm == 'TSNE':
        tSNE = TSNE(n_components=num_components, perplexity=perplexity, n_iter=2000)
        val_checkpoint_list2d = tSNE.fit_transform(all_checkpoints)
    elif algorithm == 'PCA': 
        sorted_vectors = PCA(all_checkpoints, n_components=num_components, return_vectors=True) # return_vectors return the eigenvectors
        val_checkpoint_list2d = np.matmul(all_checkpoints, sorted_vectors[:, :num_components])
        # val_checkpoint_list2d = PCA(all_checkpoints, n_components=num_components)
        # pca = PCA(n_components=num_components)
        # val_checkpoint_list2d = pca.fit_transform(all_checkpoints)
    elif algorithm == 'ICA':
        ica = FastICA(n_components=num_components)
        val_checkpoint_list2d = ica.fit_transform(all_checkpoints)
        
    color_options = ['r','g','b','m','c']
    colors = sum([[color]*n_checkpoint for n_checkpoint in n_checkpoints for color in color_options[:n_subnetworks]], [])

    first_axis = use_axes[0]
    second_axis = use_axes[1]

    if twoD:
        fig, ax = plt.subplots(ncols=len(model_names),nrows=1, figsize=(10,5))
        alg = 'PCA' if algorithm == 'PCA' else 't-SNE'
        fig.suptitle(f'{dataset}: Training trajectory for {architecture} models with {n_subnetworks} subnetworks', fontsize=15)
        axis_max = val_checkpoint_list2d.max(axis=0)
        axis_min = val_checkpoint_list2d.min(axis=0)
        span = axis_max-axis_min

        offset = 0

        ax[1].set_xlabel(f'PCA component {first_axis}', fontsize=12)
        ax[0].set_ylabel(f'PCA component {second_axis}', fontsize=12)
        for i, model in enumerate(model_names):
            # if i == 1:
            #     pass
            # else:
            ranges = [n_checkpoints[i]*n_subnetwork+offset for n_subnetwork in range(n_subnetworks+1)]
            ax[i].set_xlim([axis_min[first_axis]-0.05*span[first_axis],axis_max[first_axis]+0.05*span[first_axis]])
            ax[i].set_ylim([axis_min[second_axis]-0.05*span[second_axis],axis_max[second_axis]+0.05*span[second_axis]])
            ax[i].grid()
            
            ax[i].scatter(val_checkpoint_list2d[ranges[0]:ranges[-1],first_axis], val_checkpoint_list2d[ranges[0]:ranges[-1],second_axis], zorder=1, c=colors[offset:ranges[-1]], s=10)
            ax[i].scatter(val_checkpoint_list2d[ranges[:n_subnetworks],first_axis], val_checkpoint_list2d[ranges[:n_subnetworks], second_axis], marker='o', edgecolors='black', facecolors='none', linewidth=2, label='Initialisation', zorder=3)
            ax[i].scatter(val_checkpoint_list2d[[i-1 for i in ranges[1:n_subnetworks+1]],first_axis], val_checkpoint_list2d[[i-1 for i in ranges[1:n_subnetworks+1]], second_axis], marker='s', edgecolors='black', facecolors='none', linewidth=2, label='Endpoint', zorder=3)
            for j in range(n_subnetworks):
                ax[i].plot(val_checkpoint_list2d[ranges[j]:ranges[j+1],first_axis], val_checkpoint_list2d[ranges[j]:ranges[j+1],second_axis], label=f'subnetwork {j+1}', zorder=2, c=color_options[j], alpha=0.5)
                ax[i].set_title(f'{model} M={n_subnetworks}')
                ax[i].legend()

            offset = ranges[-1]


    else:
        fig, ax = plt.subplots(ncols=len(model_names),nrows=1, figsize=(10,5), subplot_kw=dict(projection="3d"))
        alg = 'PCA' if algorithm == 'PCA' else 't-SNE'
        fig.suptitle(f'{dataset}: Training trajectory for {architecture} models with {n_subnetworks} subnetworks', fontsize=15)
        axis_max = val_checkpoint_list2d.max(axis=0)
        axis_min = val_checkpoint_list2d.min(axis=0)
        span = axis_max-axis_min

        offset = 0

        # ax[1].set_xlabel(f'PCA component {first_axis}', fontsize=12)
        # ax[0].set_ylabel(f'PCA component {second_axis}', fontsize=12)

        for i, model in enumerate(model_names):
        
            ranges = [n_checkpoints[i]*n_subnetwork+offset for n_subnetwork in range(n_subnetworks+1)]
            ax[i].set_xlim([axis_min[0]-0.05*span[0],axis_max[0]+0.05*span[0]])
            ax[i].set_ylim([axis_min[1]-0.05*span[1],axis_max[1]+0.05*span[1]])
            ax[i].set_zlim([axis_min[2]-0.05*span[2],axis_max[2]+0.05*span[2]])
            ax[i].set_xlabel(f'PCA component 0', fontsize=12)
            ax[i].set_ylabel(f'PCA component 1', fontsize=12)
            ax[i].set_zlabel(f'PCA component 2', fontsize=12)

            ax[i].grid()

            ax[i].scatter(val_checkpoint_list2d[ranges[0]:ranges[-1],0], val_checkpoint_list2d[ranges[0]:ranges[-1],1], val_checkpoint_list2d[ranges[0]:ranges[-1],2], zorder=1, c=colors[offset:ranges[-1]])
            ax[i].scatter(val_checkpoint_list2d[ranges[:n_subnetworks],0], val_checkpoint_list2d[ranges[:n_subnetworks],1], val_checkpoint_list2d[ranges[:n_subnetworks],2], marker='o', edgecolors='black', facecolors='none', linewidth=2, label='Initialisation', zorder=3)
            ax[i].scatter(val_checkpoint_list2d[[i-1 for i in ranges[1:n_subnetworks+1]],0], val_checkpoint_list2d[[i-1 for i in ranges[1:n_subnetworks+1]],1], val_checkpoint_list2d[[i-1 for i in ranges[1:n_subnetworks+1]],2], marker='s', edgecolors='black', facecolors='none', linewidth=2, label='Endpoint', zorder=3)
            for j in range(n_subnetworks):
                ax[i].plot(val_checkpoint_list2d[ranges[j]:ranges[j+1],0], val_checkpoint_list2d[ranges[j]:ranges[j+1],1],val_checkpoint_list2d[ranges[j]:ranges[j+1],2], label=f'subnetwork {j}', zorder=2, c=color_options[j])
            
                ax[i].set_title(f'{model}')
                ax[i].legend()
                plt.grid()

            offset = ranges[-1]

    
    make_dirs(f'reports/figures/tSNE/{architecture}/{dataset}')
    plt.savefig(f'reports/figures/tSNE/{architecture}/{dataset}/{n_subnetworks}_members_tSNE_plot.png', bbox_inches='tight', dpi=600)
    plt.show()

def data_space_plot(dataset = 'CIFAR10', severity=5):
    '''
    Plot in-distribution and out-of-distribution CIFAR test data by projecting it to a 2-dimensional space. The purpose of the plot is to see how the corruption affects the data
    so to understand the behavior of models trained on the in-distribution data when tested on the out-of-distribution data.

    input:
    - dataset: Which dataset to visualize. Options: 'CIFAR10', 'CIFAR100'
    - severity: The severity of the noise on the corrupted cifar dataset
    Output:
    - None. THe function shows and saves the desired plot.
    '''
    datasets = [dataset, f'{dataset}_C']
    _, _, testdata = load_cifar10("data/") if dataset == 'CIFAR10' else load_cifar100("data/")
    testdata_C = load_CIFAR10C("data/CIFAR-10-C/", "impulse_noise", severity=severity) if dataset=='CIFAR10' else load_CIFAR100C("data/CIFAR-100-C/", "impulse_noise", severity=severity)

    #reshape corrupted CIFAR data to be in the same shape:
    inv_transform = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], std=[1/0.247, 1/0.243, 1/0.261])
    C_data = np.array([inv_transform(c_data[0]).permute(1,2,0) for c_data in testdata_C])

    #concatenate the datasets:
    all_data = np.concatenate([testdata.data[:1000], C_data[:1000]], axis=0)
    X = all_data.reshape(all_data.shape[0], -1)

    #project to 2D:
    X_pca = PCA(X, n_components = 3)

    plt.scatter(X_pca[:1000,0], X_pca[:1000, 1], label='In-distribution data')
    plt.scatter(X_pca[1000:,0], X_pca[1000:, 1], label='Out-of-distribution data')
    plt.legend()
    plt.show()
    



def plot_prediction_example(image_idx, architectures=['MediumCNN','WideResnet'], models=['MIMO'], M=3, dataset='CIFAR10', severity = 5, plot_baseline = True):
    '''
    Function to plot an image from the test dataset along with predicted probabilities for all classes.

    inputs:
    - image_idx: The index of the test image you want to visualize
    - mode: What you want to compare probabilities for. Options are 'architecture' and 'method'.
    - model_name: Name of the model which predictions you want visualised
    - M: number of subnetworks for the model given with model_name
    - dataset: Which dataset to visualise for. Choices are 'CIFAR10' and 'CIFAR100'
    - severity: The severity of corruption on the corrupted dataset.

    output:
    - no output, but the function saves a plot and shows the plot.
    '''


    #Load data:
    datasets = [dataset, f'{dataset}_C']
    _, _, testdata = load_cifar10("data/") if dataset == 'CIFAR10' else load_cifar100("data/")
    testdata_C = load_CIFAR10C("data/CIFAR-10-C/", "impulse_noise", severity=severity) if dataset=='CIFAR10' else load_CIFAR100C("data/CIFAR-100-C/", "impulse_noise", severity=severity)

    inv_transform = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], std=[1/0.247, 1/0.243, 1/0.261])
    image = testdata.data[image_idx]
    image_C = inv_transform(testdata_C[image_idx][0]).permute(1,2,0)
    images=[image,image_C]
    image_label = testdata.targets[image_idx]
    NPZs = []
    probabilities = []
    errors = []
    top10idxs = []
    n_classes = 10 if dataset =='CIFAR10' else 100
    colors = np.array(['lightblue']*n_classes, dtype='O')
    colors[image_label] = 'lightgreen'

    if dataset == 'CIFAR10':
        label_dict = {0: "airplane", 
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"}

    

    # architectures = ['MediumCNN','WideResnet']
    models_list = []
    # if 'MediumCNN' in architectures:
    #     models_list.extend([f'C_{model}' for model in models])
    # if 'WideResnet' in architectures:
    #     models_list.extend([f'C_{model}Wide' for model in models])
    
    for model in models:
        if 'MediumCNN' in architectures:
            models_list.extend([f'C_{model}'])
        if 'WideResnet' in architectures:
            models_list.extend([f'C_{model}Wide'])

    if plot_baseline:
        temp = ['Baseline']*len(architectures)
        for i in range(len(models)):
            temp.extend([models[i]]*len(architectures))
        models = temp

    n_cols = len(models)
    #load predictions

    if plot_baseline: #Make sure baseline is loaded
        if 'MediumCNN' in architectures:
            try:
                NPZs.append(np.load(f'reports/Logs/C_MIMO/{datasets[0]}/C_MIMO.npz'))
                NPZs.append(np.load(f'reports/Logs/C_MIMO/{datasets[1]}/C_MIMO_severity{severity}.npz'))
            except:
                print(f'No MediumCNN Baseline found')
        if 'WideResnet' in architectures:
            try:
                NPZs.append(np.load(f'reports/Logs/C_MIMOWide/{datasets[0]}/C_MIMOWide.npz'))
                NPZs.append(np.load(f'reports/Logs/C_MIMOWide/{datasets[1]}/C_MIMOWide_severity{severity}.npz'))
            except:
                print('No Wide ResNet Baseline Found')
    

    for model in models_list:
        try:
            NPZs.append(np.load(f"reports/Logs/{model}/{datasets[0]}/{model}.npz"))
            NPZs.append(np.load(f'reports/Logs/{model}/{datasets[1]}/{model}_severity{severity}.npz'))
        except:
            print(f"No {model} model found!")
        
    
    for i, NPZ in enumerate(NPZs):
        predictions, confidences, full_confidences, correct_preds, targets, brier_scores, NLLs = NPZ["predictions"], NPZ["confidences"], NPZ["full_confidences"], NPZ["correct_preds"], NPZ["targets_matrix"], NPZ["brier_score"], NPZ["NLL"]
        if i < 2: #Start by appending the Baseline (assumes MIMO models are loaded first)
            probabilities.append(full_confidences[:,image_idx,:,0].mean(0))
            errors.append(1.96*np.std(full_confidences[:,image_idx,:,0],axis=0)/np.sqrt(5))
        else:
            probabilities.append(full_confidences[:,image_idx,:,M-2 if models[i//2]=='MIMBO' else M-1].mean(0)) 
            errors.append(1.96*np.std(full_confidences[:,image_idx,:,M-2 if models[i//2]=='MIMBO' else M-1],axis=0)/np.sqrt(5))

        if dataset == 'CIFAR100':
            #find top 10 predictions and their probabilities
            top10idx = np.argsort(probabilities[i])[::-1][:10]
            top10idx.sort()
            probabilities[i] = probabilities[i][top10idx]
            errors[i] = errors[i][top10idx]
            top10idxs.append(top10idx)
    
    #Create plot:
    fig, ax = plt.subplots(nrows=2, ncols=n_cols+1, figsize=(np.round(4*n_cols),6)) # if plot_baseline==False else plt.subplots(nrows=2, ncols=n_cols+2, figsize=(np.round(4*(n_cols+1)),6))
    ax[0,0].set_title(f'label: {image_label}') if dataset == 'CIFAR100' else ax[0,0].set_title(f'label: {label_dict[image_label]}')
    for i, _ in enumerate(datasets):
        ax[i,0].imshow(images[i])
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])
        for j, model in enumerate(models):
            if dataset == 'CIFAR10':
                ax[i,j+1].bar(x=np.arange(0,10), height = probabilities[i+(j*2)], color=colors)
                ax[i,j+1].errorbar(x=np.arange(0,10), y = probabilities[i+(j*2)], yerr=errors[i+(j*2)], color='black', fmt='none', capsize=3)
                ax[i,j+1].set_ylim(0,1)
                ax[i,j+1].set_xticks(np.arange(0,10),list(label_dict.values()), rotation=60, size=8)
                # ax[i,j+1].tick_params(labelrotation=60)
            elif dataset == 'CIFAR100':
                ax[i,j+1].bar(x=np.arange(0,10), height = probabilities[i+(j*2)], color=colors[top10idxs[i+(j*2)]])
                ax[i,j+1].errorbar(x=np.arange(0,10), y = probabilities[i+(j*2)], yerr=errors[i+(j*2)], color='black', fmt='none', capsize=3)
                ax[i,j+1].set_ylim(0,1)
                ax[i,j+1].set_xticks(np.arange(0,10), list(top10idxs[i+(j*2)]))
            if i == 0:
                # ax[i,j+1].set_title(f'Model: {models[j%len(models)]} \n Architecture: {architectures[(j*int(len(architectures)/len(models)))//len(architectures)]}')
                ax[i,j+1].set_title(f'Model: {models[j%len(models)]} \n Architecture: {architectures[j%len(architectures)]}')
            if j > 0:
                ax[i,j+1].set_yticks([])
    make_dirs('reports/figures/prediction_examples')
    plt.savefig(f'reports/figures/prediction_examples/prediction_example_{image_idx}_M{M}_subnetworks_{dataset}.png', bbox_inches='tight', dpi=900)
    fig.tight_layout()
    plt.show()

def plot_prediction_distribution(architectures = ['MediumCNN'], models = ['MIMO'], M = 3, dataset = 'CIFAR10', severity = 5, plot_baseline=True):

    datasets = [dataset, f'{dataset}_C']
    NPZs = []
    predicted_classes = []
    errors = []
    true_positives = []
    n_classes = 10 if dataset =='CIFAR10' else 100
    colors = np.array(['tomato']*n_classes, dtype='O')
    tp_colors = np.array(['lightgreen']*n_classes, dtype='O')

    if dataset == 'CIFAR10':
        label_dict = {0: "airplane", 
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"}
    
    models_list = []
    for model in models:
        if 'MediumCNN' in architectures:
            models_list.extend([f'C_{model}'])
        if 'WideResnet' in architectures:
            models_list.extend([f'C_{model}Wide'])
    
    if plot_baseline:
        
        temp = ['Baseline']*len(architectures)
        for i in range(len(models)):
            temp.extend([models[i]]*len(architectures))
        models = temp
    n_cols = len(models)

    #load predictions

    if plot_baseline: #Make sure baseline is loaded
        if 'MediumCNN' in architectures:
            try:
                NPZs.append(np.load(f'reports/Logs/C_MIMO/{datasets[0]}/C_MIMO.npz'))
                NPZs.append(np.load(f'reports/Logs/C_MIMO/{datasets[1]}/C_MIMO_severity{severity}.npz'))
            except:
                print(f'No MediumCNN Baseline found')
                return
        if 'WideResnet' in architectures:
            try:
                NPZs.append(np.load(f'reports/Logs/C_MIMOWide/{datasets[0]}/C_MIMOWide.npz'))
                NPZs.append(np.load(f'reports/Logs/C_MIMOWide/{datasets[1]}/C_MIMOWide_severity{severity}.npz'))
            except:
                print('No Wide ResNet Baseline Found')
                return
    
    for model in models_list:
        try:
            NPZs.append(np.load(f"reports/Logs/{model}/{datasets[0]}/{model}.npz"))
            NPZs.append(np.load(f'reports/Logs/{model}/{datasets[1]}/{model}_severity{severity}.npz'))
        except:
            print(f"No {model} model found!")
            return
        
    
    for i, NPZ in enumerate(NPZs):
        predictions, confidences, full_confidences, correct_preds, targets, brier_scores, NLLs = NPZ["predictions"], NPZ["confidences"], NPZ["full_confidences"], NPZ["correct_preds"], NPZ["targets_matrix"], NPZ["brier_score"], NPZ["NLL"]
        if i < len(architectures)*2: #Start by appending the Baseline (assumes MIMO models are loaded first)
            # unique, counts = np.unique(predictions[0,:,0], return_counts=True)
            counts = np.array([np.array([(predictions[j,:,0]==c).sum() for c in range(n_classes)]) for j in range(5)])
            true_positive = np.array([np.array([np.sum(correct_preds[j,np.where(predictions[j,:,0]==c),0]) for c in range(n_classes)]) for j in range(5)]) 


        else:
            # unique, counts = np.unique(predictions[0,:,M-1 if models[i//2]=='MIMBO' else M-1], return_counts=True)
            counts = np.array([np.array([(predictions[j,:,M-2 if models[i//2]=='MIMBO' else M-1]==c).sum() for c in range(n_classes)]) for j in range(5)])
            true_positive = np.array([np.array([np.sum(correct_preds[j, np.where(predictions[j,:,M-2 if models[i//2]=='MIMBO' else M-1]==c), M-2 if models[i//2]=='MIMBO' else M-1]) for c in range(n_classes)]) for j in range(5)]) 


        predicted_classes.append(counts.mean(0))
        errors.append(1.96*np.std(counts,axis=0)/np.sqrt(5))
        true_positives.append(true_positive.mean(0))

    fig, ax = plt.subplots(nrows=2, ncols=n_cols , figsize=(np.round(3*n_cols),6))# if plot_baseline==False else plt.subplots(nrows=2, ncols=n_cols+len(architectures), figsize=(np.round(4*(n_cols+1)),6))
    for i, _ in enumerate(datasets):
        for j, model in enumerate(models):
            if dataset == 'CIFAR10':
                ax[i,j].bar(x=np.arange(0,10), height = predicted_classes[i+(j*2)], color=colors, label = 'False positive')
                ax[i,j].bar(x=np.arange(0,10), height = true_positives[i+(j*2)], color=tp_colors, label = 'True positive')
                ax[i,j].errorbar(x=np.arange(0,10), y = predicted_classes[i+(j*2)], yerr=errors[i+(j*2)], color='black', fmt='none', capsize=3)
                ax[i,j].set_ylim(0,np.array(predicted_classes).max()+np.array(errors).max())
                ax[i,j].set_xticks(np.arange(0,10),list(label_dict.values()), rotation=60, size=8)
                
                
                # ax[i,j+1].tick_params(labelrotation=60)
            elif dataset == 'CIFAR100':
                ax[i,j].bar(x=np.arange(0,100), height = predicted_classes[i+(j*2)], color=colors, label='False Positives')
                ax[i,j].bar(x=np.arange(0,100), height = true_positives[i+(j*2)], color=tp_colors, label='True Positives')
                ax[i,j].set_ylim(0,np.array(predicted_classes).max()+np.array(errors).max())
                ax[i,j].set_xticks([])
            if i == 0:
                # ax[i,j].set_title(f'Model: {models[j%len(models)]} \n Architecture: {architectures[(j*int(len(architectures)/(len(models)/2)))%len(architectures)]}')
                ax[i,j].set_title(f'Model: {models[j%len(models)]} \n Architecture: {architectures[j%len(architectures)]}')
            if j > 0:
                ax[i,j].set_yticks([])
    ax[0,n_cols-1].legend()
    fig.tight_layout()
    make_dirs('reports/figures/prediction_distribution')
    plt.savefig(f'reports/figures/prediction_distribution/{dataset}_M{M}.png', bbox_inches='tight', dpi=600)
    plt.show()

def kl_weighting_plot(csv_path):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    df = pd.read_csv(csv_path)

    y1 = np.asarray(list(df[df.columns[1]]))
    y1 = y1[~np.isnan(y1)]
    y2 = np.asarray(list(df[df.columns[4]]))
    y2 = y2[~np.isnan(y2)]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(y1, label=r'$\beta_b = \frac{2^{B-b}}{2^B-1}$')
    ax.plot(y2, label=r'$\beta_b = \frac{n_b}{N}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    print("hej")

def compute_disagreement(subnetwork_predictions):
    n_samples = subnetwork_predictions.shape[0]
    n_subnetworks = subnetwork_predictions.shape[1]

    # Compute the disagreement between all pairs of subnetworks
    disagreement_matrix = np.zeros((n_subnetworks, n_subnetworks))
    for i in range(n_subnetworks):
        for j in range(n_subnetworks):
            if i != j:
                disagreement_matrix[i, j] = (subnetwork_predictions[:, i] != subnetwork_predictions[:, j]).float().mean().item()
    
    disagreement = 1/(n_subnetworks * (n_subnetworks -1))*disagreement_matrix.sum()

    return disagreement

def compute_KL_divergence(checkpoint):
    n_samples = checkpoint.shape[0]
    n_subnetworks = checkpoint.shape[2]

    # Compute the KL divergence between all pairs of subnetworks
    KL_divergence_matrix = np.zeros((n_subnetworks, n_subnetworks))
    for i in range(n_subnetworks):
        for j in range(n_subnetworks):
            if i != j:
                KL_divergence_matrix[i, j] = torch.nn.functional.kl_div(checkpoint[:,:, i], checkpoint[:, :, j], reduction='mean', log_target=True).item()
    
    average_divergence = 1/(n_subnetworks * (n_subnetworks -1)) * KL_divergence_matrix.sum()

    return average_divergence

def make_barplot(checkpoint, subnetworks):

    # Calculate the average of all log probabilities
    avg_log_probs = torch.exp(checkpoint).mean(dim=0).cpu().detach().numpy()
    colours = ['r','g','b','m','c']

    fig, ax = plt.subplots()

    X = np.arange(len(avg_log_probs))

    # Repeat the process three times

    # Plot the bar plot with semitransparent bars

    for i in range(subnetworks):
        ax.bar(X + 0.15*i - (subnetworks//2)*0.15, avg_log_probs[:,i], alpha=1, color=colours[i], label=f'Subnetwork {i+1}', width=0.15)
    
    # Add the legend
    plt.legend()
    plt.xlabel("Classes")
    plt.xticks(X)
    plt.ylabel("Confidence")

    # Show the plot
    plt.show()