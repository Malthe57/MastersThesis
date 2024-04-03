import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_single, function_space_plots

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
    per_rep_std = np.std(accuracies, axis=0, ddof=1) # std of repetitions

    return per_rep_accuracy, per_rep_std


if __name__ == '__main__':

    dataset = "CIFAR10"

    models = ["C_MIMO", "C_BNN", "C_MIMBO"]

    Ms = [3]

    for model in models:
        print("Visualizing model:", model)
        try:
            NPZ = np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        except:
            print(f"No {model} model found!")
        else:
            predictions, confidences, full_confidences, correct_preds, targets, brier_scores = NPZ["predictions"], NPZ["confidences"], NPZ["full_confidences"], NPZ["correct_preds"], NPZ["targets_matrix"], NPZ["brier_score"]
            per_rep_accuracy, per_rep_std = model_accuracy(correct_preds)
            rep_idxs = get_rep_idxs(correct_preds)
            for i in range(rep_idxs.shape[0]):
                if "BNN" in model:
                    reliability_plot_classification_single(correct_predictions=correct_preds[rep_idxs[i], :], confidence=confidences[rep_idxs[i],:], model_name=model)
                    print(f"{model} test accuracy: {per_rep_accuracy} \pm {1.96*per_rep_std} \n")
                else:
                    reliability_plot_classification_single(correct_predictions=correct_preds[rep_idxs[i], i, :], confidence=confidences[rep_idxs[i], i,:], model_name=model, M=i+2)
                    print(f"{model} M{i+2} test accuracy: {per_rep_accuracy[i]} \pm {1.96*per_rep_std[i]} \n")

        try:
            for M in Ms:
                checkpoint = torch.load(f'models/classification/checkpoints/{model}/{dataset}/M{M}/{model}_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))
        except:
            print("You Failed!")
        else:
            function_space_plots(checkpoint, f'{model}_{M}_members')
    # try:
    #     MIMOs = np.load(f"reports/Logs/C_MIMO/{dataset}/C_MIMO.npz")
    # except:
    #     print("No MIMO models found!")
    #     MIMO = False
    # else:
    #     MIMO = True
    #     MIMO_predictions, MIMO_confidences, MIMO_full_confidences, MIMO_correct_preds, MIMO_targets, MIMO_brier_scores = MIMOs["predictions"], MIMOs["confidences"], MIMOs["full_confidences"], MIMOs["correct_preds"], MIMOs["targets_matrix"], MIMOs["brier_scores"]
    
    # try:
    #     Naives = np.load(f"reports/Logs/C_Naive/{dataset}/C_Naive.npz")
    # except:
    #     print("No naive models found!")
    #     Naive = False
    # else:
    #     Naive = True
    #     Naive_predictions, Naive_confidences, Naive_full_confidences, Naive_correct_preds, Naive_targets, Naive_brier_scores = Naives["predictions"], Naives["confidences"], MIMOs["full_confidences"], MIMOs["correct_preds"], MIMOs["targets_matrix"], MIMOs["brier_scores"]

    # try:
    #     BNNs = np.load(f"reports/Logs/C_BNN/{dataset}/C_BNN.npz")
    # except:
    #     print("No bayesian model found!")
    #     BNN = False
    # else:
    #     BNN = True
    #     BNN_predictions, BNN_confidences, BNN_full_confidences, BNN_correct_preds, BNN_targets, BNN_brier_scores = BNNs["predictions"], BNNs["confidences"], BNNs["full_confidences"], BNNs["correct_preds"], BNNs["targets_matrix"], BNNs["brier_scores"]
    
    
    # try:
    #     MIMBOs = np.load(f"reports/Logs/C_MIMBO/{dataset}/C_MIMBO.npz")
    # except:
    #     print("NO MIMBO model found")
    #     MIMBO = False
    # else:
    #     MIMBO = True
    #     MIMBO_predictions, MIMBO_confidences, MIMBO_full_confidences, MIMBO_correct_preds, MIMBO_targets, MIMBO_brier_scores = MIMBOs["predictions"], MIMBOs["confidences"], MIMBOs["full_confidences"], MIMBOs["correct_preds"], MIMBOs["targets_matrix"], MIMBOs["brier_scores"]
    
