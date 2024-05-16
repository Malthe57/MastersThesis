import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_single, function_space_plots, multi_function_space_plots

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

if __name__ == '__main__':

    dataset = "CIFAR10"

    models = ["C_BNN"]

    Ms = [1]

    for model in models:
        print("Visualizing model:", model)
        try:
            NPZ = np.load(f"reports/Logs/{model}/{dataset}/{model}.npz")
        except:
            print(f"No {model} model found!")
        else:
            predictions, confidences, full_confidences, correct_preds, targets, brier_scores, NLLs = NPZ["predictions"], NPZ["confidences"], NPZ["full_confidences"], NPZ["correct_preds"], NPZ["targets_matrix"], NPZ["brier_score"], NPZ["NLL"]
            for M in Ms:
                if "BNN" in model:
                    per_rep_accuracy, per_rep_SE = model_accuracy(correct_preds)   
                    reliability_plot_classification_single(correct_predictions=correct_preds, confidence=confidences, model_name=model)
                    print(f"{model} test accuracy: {per_rep_accuracy} \pm {1.96*per_rep_SE} \n")
                else:
                    per_rep_accuracy, per_rep_SE = model_accuracy(correct_preds[:,:,M-2])   
                    reliability_plot_classification_single(correct_predictions=correct_preds[:,:,M-2], confidence=confidences[:,:,M-2], model_name=model, M=M)
                    print(f"{model} M{M} test accuracy: {per_rep_accuracy} \pm {1.96*per_rep_SE} \n")
    
    
                

        # try:
        #     for M in Ms:
        #         checkpoint = torch.load(f'models/classification/checkpoints/{model}/{dataset}/M{M}/{model}_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))
        # except:
        #     print(f"Checkpoints for {model} not found!")
        # else:
        #     function_space_plots(checkpoint, f'{model}_{M}_members')
    
    # try:
    #     for M in Ms:
    #         checkpoint_list = []
    #         checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMOWide/{dataset}/M{M}/C_MIMOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:100,:,:,:])
    #         checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_NaiveWide/{dataset}/M{M}/C_NaiveWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:100,:,:,:])
    #         checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBOWide/{dataset}/M{M}/C_MIMBOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[:100,:,:,:])
    # except:
    #     print('Try again loser >:)')
    # else:
    #     # for i in range(1, 50):
    #     multi_function_space_plots(checkpoint_list, ['C_MIMO','C_Naive','C_MIMBO'], n_samples=5, perplexity=15, n_components=10, algorithm='TSNE')
    # # try:
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
    
