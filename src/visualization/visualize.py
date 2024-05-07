import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.utils import make_dirs

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

def reliability_plot_classification_single(correct_predictions, confidence, model_name, M=1):
        #Code for generating reliability diagram:
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4,4))

    reps = correct_predictions.shape[0]
    linspace = np.arange(0, 1.1, 0.1)
    bins_range = np.quantile(correct_predictions.flatten(), linspace)
    n_samples = len(correct_predictions.T)
    
    conf_step_height = np.zeros((reps, 10))
    acc_step_height = np.zeros((reps,10))

    lengths = np.zeros((reps, 10))
    ECEs = np.zeros((reps, 10))
    for j in range(reps):
        for i in range(10):
            loc = np.where(np.logical_and(confidence[j,:]>=bins_range[i], confidence[j,:]<bins_range[i+1]))[0]
            if correct_predictions[j,loc].shape[0] != 0:
                acc_step_height[j, i] = np.mean(correct_predictions[j, loc])
                conf_step_height[j, i] = np.mean(confidence[j, loc])
                lengths[j, i] = correct_predictions[j,loc].shape[0]
                ECEs[j,i] = np.abs(acc_step_height[j, i]-conf_step_height[j, i])*lengths[j,i]
    
    ECE = np.sum(ECEs)/n_samples
    # MSE_step_std = MSE_step_height[MSE_step_height!=0].std(axis=0)
    acc_step_std = np.zeros(10)
    acc_final_step = np.zeros(10)
    
    for j, values in enumerate(acc_step_height.T):
        if np.all(np.array(values) == 0):
            acc_step_std[j] = 0
            acc_final_step[j] = 0
        else:
            acc_step_std[j] = np.std(values[values!=0])
            acc_final_step[j] = np.mean(values[values!=0])

    acc_step_ub = acc_final_step + 1.96*acc_step_std
    acc_step_lb = acc_final_step - 1.96*acc_step_std
    
    ECE = np.sum(ECEs)/n_samples
    if M>1:
        print(f"{model_name} M{M} ECE: {ECE}")
    else:
        print(f"{model_name} ECE: {ECE}")
    
    fig.supxlabel("Confidence")
    fig.supylabel("Accuracy")
    fig.suptitle(f'Reliability Diagram')
    # fig.set_layout_engine('compressed')
    
    ax.grid(linestyle='dotted', zorder=0)
    ax.stairs(acc_final_step, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    ax.stairs(acc_step_ub, bins_range, baseline = acc_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='CI upper bound', zorder=2)
    ax.stairs(acc_step_lb, bins_range, baseline = acc_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label= 'CI lower bound', zorder=2)
    # ax.stairs(conf_step_height, bins_range, baseline = acc_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
    ax.plot(linspace, linspace, linestyle='--', color='gray', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.text(0.1, 0.6, f'ECE={np.round(ECE,5)}', backgroundcolor='lavender', alpha=1.0, fontsize=8.0)

    if M>1:
        ax.set_title(f"{model_name}_M{M}")
        plt.savefig(f"reports/figures/reliability_diagrams/classification/{model_name}_M{M}_reliability_diagram.png")
    else:
        ax.set_title(f"{model_name}")
        plt.savefig(f"reports/figures/reliability_diagrams/classification/{model_name}_reliability_diagram.png")
    plt.show()

def reliability_diagram_regression(predictions, targets, predicted_std, M, model_name):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    
    reps = predictions.shape[0]
    predictions = predictions
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
    
    ECE = np.sum(ECEs)/n_samples
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
            
    
    
    Variance_step_std = Variance_step_height[Variance_step_height!=0].std(axis=0)
    # Variance_step_height = Variance_step_height[Variance_step_height!=0].mean(axis=0)

    MSE_step_ub = MSE_final_step + 1.96*MSE_step_std
    MSE_step_lb = MSE_final_step - 1.96*MSE_step_std


    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linestyle='dotted', zorder=0)
    plt.stairs(MSE_final_step, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    plt.stairs(MSE_step_ub, bins_range, baseline = MSE_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='CI upper bound', zorder=2)
    plt.stairs(MSE_step_lb, bins_range, baseline = MSE_final_step, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label= 'CI lower bound', zorder=2)
    plt.plot(bins_range, bins_range, linestyle='--', color='gray', zorder=3)
    plt.legend()
    plt.title(f"Regression reliability plot for {model_name} with M={M}")

    plt.xlabel("Predicted variance") 
    plt.ylabel("Mean squared error") 
    

    if M > 1:
        plt.title(f"Regression reliability plot for {model_name} with M={M}")
        plt.savefig(f"reports/figures/reliability_diagrams/regression/{model_name}_{M}_reliability_diagram.png")  
        print(f'ECE for {model_name} with {M} members: {ECE}') 
    else:
        plt.title(f"Regression reliability plot for {model_name}")
        plt.savefig(f"reports/figures/reliability_diagrams/regression/{model_name}_reliability_diagram.png")
        print(f'ECE for {model_name}: {ECE}')  

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

def multi_function_space_plots(checkpoints_list, model_names, n_samples=20, perplexity=10):
    '''
    Inputs:
    - Checkpoints_list: a list containing loaded checkpoint. It is assumed that the models listed have the same number of checkpoints, subnetworks and predicted classes
    - model_names: list of names used for plot title
    - n_samples: number of predictions used at each checkpoint. Caps out at the number of predictions saved in the checkpoint while training. Maximum n_samples should be the same for all models
    '''
    _ , max_samples, n_classes, n_subnetworks, = checkpoints_list[0].shape
    n_checkpoints = [checkpoints.shape[0] for checkpoints in checkpoints_list]
    if n_samples > max_samples:
        print(f'the n_samples parameter is too large, reducing to the max value of {max_samples}')
        n_samples = max_samples

    # concatenate checkpoints for different models in dim 0. Cap it to n_samples
    checkpoints_concat = torch.concat(([c for c in checkpoints_list])).numpy()[:, :n_samples, :, :]
    # reshape to [n_checkpoints*n_subnetworks, n_samples, n_classes] and then to [n_checkpoints*n_subnetworks, -1]
    all_checkpoints = checkpoints_concat.reshape((-1, n_samples, n_classes)).reshape((sum(n_checkpoints)*n_subnetworks, -1))

    # fit t-SNE to checkpoints
    # tSNE = TSNE(n_components=2, perplexity=perplexity, n_iter=2000)
    # val_checkpoint_list2d = tSNE.fit_transform(all_checkpoints)
    pca = PCA(n_components=2)
    val_checkpoint_list2d = pca.fit_transform(all_checkpoints)

    color_options = ['r','g','b','y','c']
    colors = sum([[color]*n_checkpoint for n_checkpoint in n_checkpoints for color in color_options[:n_subnetworks]], [])

    fig, ax = plt.subplots(ncols=len(model_names),nrows=1, figsize=(10,5))
    fig.suptitle(f't-SNE plot of subnetwork predictions for models with {n_subnetworks} members')

    offset = 0
    for i, model in enumerate(model_names):
    
        ranges = [n_checkpoints[i]*n_subnetwork+offset for n_subnetwork in range(n_subnetworks+1)]
        if i == 3:
            pass
        else:

            ax[i].scatter(val_checkpoint_list2d[ranges[0]:ranges[-1],0], val_checkpoint_list2d[ranges[0]:ranges[-1],1], zorder=1, c=colors[offset:ranges[-1]])
            ax[i].scatter(val_checkpoint_list2d[ranges[:n_subnetworks],0], val_checkpoint_list2d[ranges[:n_subnetworks],1], marker='o', edgecolors='black', facecolors='none', linewidth=2, label='Initialisation', zorder=3)
            for j in range(n_subnetworks):
                ax[i].plot(val_checkpoint_list2d[ranges[j]:ranges[j+1],0], val_checkpoint_list2d[ranges[j]:ranges[j+1],1], label=f'subnetwork {j}', zorder=2, c=color_options[j])
                ax[i].grid()
                ax[i].set_title(f'{model}')
                ax[i].legend()
        
        offset = ranges[-1]
    
    make_dirs(f'reports/figures/tSNE/')
    plt.savefig(f'reports/figures/tSNE/{n_subnetworks}_members_tSNE_plot.png')
    plt.show()


