import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_single, function_space_plots, multi_function_space_plots, plot_prediction_example, data_space_plot

def get_rep_idxs(correct_preds_matrix : torch.tensor):
    """
    Returns the idx of the repitition with the highest accuracy

    input:
        correct_preds-matrix: torch.tensor
    output: 
        torch.tensor (containing indices)
    """
    rep_idx = correct_preds_matrix.mean(-1).argmax(0)
    return rep_idx if len(correct_preds_matrix.shape) == 3 else np.array([rep_idx]) # mean over test points dim, then argmax over rep dim

def model_accuracy(correct_preds_matrix : torch.tensor):
    """
    Returns the accuracy of a model

    input:
        correct_preds-matrix: torch.tensor
    output: 
        per_rep_accuracy: torch.tensor
        per_rep_std: torch.tensor
    """

    accuracies = np.mean(correct_preds_matrix, axis=-1) # mean over datapoints
    per_rep_accuracy = np.mean(accuracies, axis=0) # mean of repetitions
    per_rep_SE = np.std(accuracies, axis=0) / np.sqrt(accuracies.shape[0]) # standard error of the mean of repetitions

    return per_rep_accuracy, per_rep_SE

def accuracy_and_ECE():
    dataset = "CIFAR10"

    models = ["C_MIMOWide"]

    severity = None

    Ms = [1,2,3,4,5]

    for model in models:
        print("Visualizing model:", model)
        try:
            NPZ = np.load(f"reports/Logs/{model}/{dataset}/{model}_severity{severity}.npz") if severity else np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        except:
            print(f"No {model} model found!")
        else:
            predictions, confidences, full_confidences, correct_preds, targets, brier_scores, NLLs = NPZ["predictions"], NPZ["confidences"], NPZ["full_confidences"], NPZ["correct_preds"], NPZ["targets_matrix"], NPZ["brier_score"], NPZ["NLL"]
            for i, M in enumerate(Ms):
                if "BNN" in model:
                    per_rep_accuracy, per_rep_SE = model_accuracy(correct_preds)   
                    reliability_plot_classification_single(correct_predictions=correct_preds, confidence=confidences, model_name=model, dataset=dataset, severity=severity)
                    print(f"{model} test accuracy: {per_rep_accuracy} \pm {1.96*per_rep_SE} \n")
                else:
                    per_rep_accuracy, per_rep_SE = model_accuracy(correct_preds[:,:,i])
                    if M == 1:
                        reliability_plot_classification_single(correct_predictions=correct_preds[:,:,i], confidence=confidences[:,:,i], model_name='C_BaselineWide' if model[-4:] == 'Wide' else 'C_Baseline', dataset=dataset, M=M, severity=severity)
                    else:
                        reliability_plot_classification_single(correct_predictions=correct_preds[:,:,i], confidence=confidences[:,:,i], model_name=model, dataset=dataset, M=M, severity=severity)
                    print(f"{model} M{M} test accuracy: {per_rep_accuracy} \pm {1.96*per_rep_SE} \n")

def function_space():
    try:
        Ms = [3]
        dataset = "CIFAR10"
        for M in Ms:
            checkpoint_list = []
            checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMOWide/{dataset}/M{M}/C_MIMOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:,:,:,:])
            checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_NaiveWide/{dataset}/M{M}/C_NaiveWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:,:,:,:])
            checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBOWide/{dataset}/M{M}/C_MIMBOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:,:,:,:])
    except:
        print('Try again loser >:)')
    else:
        multi_function_space_plots(checkpoint_list, ['C_MIMO','C_Naive','C_MIMBO'], n_samples=5, perplexity=15, num_components=3, algorithm='PCA')
    
def plot_prediction_example():
    Ms = [3]
    dataset = "CIFAR10"
    # plot_prediction_example(4, architectures=['MediumCNN','WideResnet'], models=['MIMO','MIMBO'], M=4, dataset=dataset, severity=5)
    data_space_plot(dataset=dataset, severity=1)

if __name__ == '__main__':

    accuracy_and_ECE()



