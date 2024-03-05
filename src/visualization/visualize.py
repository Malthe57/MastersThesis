import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd

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
    # plt.show()


def plot_log_probs(log_priors, log_variational_posteriors, NLLs):

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

    plt.savefig(f"reports/figures/losses/regression/BNN/BNN_log_probs.png")
    # plt.show()

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

def reliability_plot_classification(correct_predictions, confidence, naive_correct_predictions, naive_confidence, model_name, naive_model_name, M):
        #Code for generating reliability diagram:
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8,4))

    bins_range = np.arange(0, 1.1, 0.1)
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

def reliability_plot_classification_single(correct_predictions, confidence, model_name):
        #Code for generating reliability diagram:
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4,4))

    bins_range = np.arange(0, 1.1, 0.1)
    n_samples = len(correct_predictions)

    conf_step_height = np.zeros(10)
    acc_step_height = np.zeros(10)
    ECE_values = np.zeros(10)
    for i in range(len(bins_range)-1):
        loc = np.where(np.logical_and(confidence>=bins_range[i], confidence<bins_range[i+1]))
        conf_step_height[i] = np.mean(confidence[loc[0]])
        acc_step_height[i] = np.mean(correct_predictions[loc[0]])
        if np.isnan(conf_step_height[i]) == False:
            ECE_values[i] = len(loc[0])/n_samples*np.abs(acc_step_height[i]-conf_step_height[i])
        else:
            acc_step_height[i] = 0.0
    
    ECE = np.sum(ECE_values)/n_samples
    print(f"{model_name} ECE: {ECE}")

    
    
    fig.supxlabel("Confidence")
    fig.supylabel("Accuracy")
    fig.suptitle(f'Reliability Diagram')
    fig.set_layout_engine('compressed')
    
    ax.grid(linestyle='dotted', zorder=0)
    ax.stairs(acc_step_height, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    ax.stairs(conf_step_height, bins_range, baseline = acc_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
    ax.plot(bins_range, bins_range, linestyle='--', color='gray', zorder=3)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{model_name}")
    ax.legend()
    ax.text(0.7, 0.05, f'ECE={np.round(ECE,5)}', backgroundcolor='lavender', alpha=1.0, fontsize=8.0)

    plt.savefig(f"reports/figures/{model_name}_reliability_diagram.png")
    plt.show()

def reliability_diagram_regression(predictions, targets, predicted_std, M, model_name):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    
    predictions = predictions.flatten()
    predicted_variance = (predicted_std**2).flatten()
    # make bins from 
    bins_range = np.linspace(0, np.max(predicted_variance), 11)
    n_samples = len(predictions)

    MSE_step_height = np.zeros(10)
    Variance_step_height = np.zeros(10)

    squared_error = np.power(predictions - targets, 2)

    for i in range(10):
        loc = np.where(np.logical_and(predicted_variance>=bins_range[i], predicted_variance<bins_range[i+1]))[0]
        MSE_step_height[i] = np.mean(squared_error[loc])
        Variance_step_height[i] = np.mean(predicted_variance[loc])

    plt.grid(linestyle='dotted', zorder=0)
    plt.stairs(MSE_step_height, bins_range, fill = True, color='b', edgecolor='black', linewidth=3.0, label='Outputs', zorder=1)
    plt.stairs(Variance_step_height, bins_range, baseline = MSE_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
    plt.plot(bins_range, bins_range, linestyle='--', color='gray', zorder=3)
    plt.legend()
    plt.title(f"Regression reliability plot for {model_name} with M={M}")

    plt.xlabel("Predicted variance") 
    plt.ylabel("Mean squared error") 

    plt.savefig(f"reports/figures/{model_name}_{M}_reliability_diagram.png")   

    plt.show()

    
def function_space_plots(model_name):
    checkpoint_list = torch.load(f'models/{model_name}.pt')
    checkpoint_list = torch.stack(checkpoint_list[:,:5,:]).flatten()
    tSNE = TSNE(checkpoint_list.shape)
    val_checkpoint_list2d = tSNE.fit_transform(checkpoint_list)
    plt.plot(checkpoint_list)

