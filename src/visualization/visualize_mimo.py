import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd

def plot_loss(losses, val_losses, model_name="MIMO"):

    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle(f"{model_name} Losses")

    ax[0].plot(losses, label='Train loss')
    ax[0].set_title('Train loss')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')

    ax[1].plot(val_losses, label='Validation loss', color='orange')
    ax[1].set_title('Validation loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')
    plt.savefig(f"reports/figures/{model_name}_losses.png")  
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
    plt.savefig(f"reports/figures/{model_name}_{mode}_weights")
    plt.show()

def plot_regression(x_test, y_test, mimo_pred_matrix, mimo_stds, Ms):
    # plot data
    fig, ax = plt.subplots(1,1, figsize=(18,12))

    ### plot mimo ##
    ax.grid()
    # ax[0].plot(x_test, line, '--', label='true function', color='red')
    # plot test data
    ax.plot(x_test, y_test, '.', label='Test data', color='black')

    # plot predicitons with confidence intervals
    for i in range(len(Ms)):
        ax.plot(x_test, mimo_pred_matrix[i], '-', label=f'Mean MIMO Predictions with {Ms[i]} members', linewidth=2)
        ax.fill_between(x_test, mimo_pred_matrix[i] - 1.96*mimo_stds[i], mimo_pred_matrix[i] + 1.96*mimo_stds[i], alpha=0.2, label=f'Confidence Interval with {Ms[i]} members')

    ax.legend()

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
    plt.savefig(f"reports/figures/{model_name}_{M}_confidence_plots.png")
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
    plt.stairs(bins_range[1:], bins_range, baseline = MSE_step_height, hatch="/", fill = True, alpha=0.3, color='r', edgecolor='r', linewidth=3.0, label='Gap', zorder=2)
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


if __name__ == '__main__':
    #init params
    modes = ['regression','classification']
    mode = modes[1]
    model_names = ['MIMO','Naive', 'VarMIMO', 'C_MIMO', 'C_Naive']
    model_name = model_names[3]
    naive_model_name = "C_Naive"
    base_load_path = 'reports/Logs/'
    load_path = os.path.join(base_load_path, model_name)
    naive_load_path = os.path.join(base_load_path, naive_model_name)
    Ms = [2,3,4]
    # Ms = [3]
    N_test = 10000
    # M = Ms[1]

    if mode == 'regression':
        test_df = pd.read_csv('data/toydata/test_data.csv')
        targets = np.array(list(test_df['y']))
        for M in Ms:
            #load data
            outputs = np.load("reports/Logs/VarMIMO/VarMIMO_3.npz")
            
            predictions = outputs['predictions']
            
            predicted_variance = outputs['predicted_variance']
            
            reliability_diagram_regression(predictions, targets, predicted_variance, model_name=model_name, M=M)


    elif mode == 'classification':
        for M in Ms:
            #load data
            predictions = np.load(os.path.join(base_load_path,f'{model_name}/M{M}_predictions.npy')) 
            confidences = np.load(os.path.join(base_load_path,f'{model_name}/M{M}_confidences.npy')) 
            correct_predictions = np.load(os.path.join(base_load_path,f'{model_name}/M{M}_correct_predictions.npy')) 

            Naive_predictions = np.load(os.path.join(base_load_path,f'{naive_model_name}/Naive_M{M}_predictions.npy')) 
            Naive_confidences = np.load(os.path.join(base_load_path,f'{naive_model_name}/Naive_M{M}_confidences.npy')) 
            Naive_correct_predictions = np.load(os.path.join(base_load_path,f'{naive_model_name}/Naive_M{M}_correct_predictions.npy'))


            reliability_plot_classification(correct_predictions=correct_predictions, confidence = confidences, naive_correct_predictions=Naive_correct_predictions, naive_confidence=Naive_confidences, model_name=model_name, naive_model_name=naive_model_name, M=M)

    
