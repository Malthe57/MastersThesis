import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_single, function_space_plots, multi_function_space_plots, plot_prediction_example, data_space_plot, plot_prediction_distribution, kl_weighting_plot, compute_disagreement, compute_KL_divergence

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
    dataset = "CIFAR10_C"

    models = ["C_BNN"]

    severity = 5

    Ms = [1]

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

def function_space(Ms=[3], datasets=['CIFAR10'], is_resnet = True, use_axes=[1,2], twoD=False, rep=1):
    for dataset in datasets:
        for M in Ms:
            try:
                checkpoint_list = []
                if is_resnet:
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMOWide/{dataset}/M{M}/C_MIMOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_NaiveWide/{dataset}/M{M}/C_NaiveWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBOWide/{dataset}/M{M}/C_MIMBOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    architecture = 'Wide ResNet'
                else:
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMO/{dataset}/M{M}/C_MIMO_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[-1:])
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_Naive/{dataset}/M{M}/C_Naive_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu'))[-1:])
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBO/{dataset}/M{M}/C_MIMBO_rep1_checkpoints.pt', map_location=torch.device('cpu'))[-1:])
                    architecture = 'MediumCNN'
            except:
                print('Try again loser >:)')
            else:
                multi_function_space_plots(checkpoint_list, [f'MIMO',f'Naive',f'MIMBO'], dataset=dataset, architecture=architecture, n_samples=256, perplexity=15, num_components=3, use_axes=use_axes, algorithm='PCA', twoD=twoD)
    
def plot_example():
    Ms = [3]
    dataset = "CIFAR10"
    for i in range(10):
        plot_prediction_example(i+9, architectures=['MediumCNN','WideResnet'], models=['MIMBO'], M=3, dataset=dataset, severity=5, plot_baseline=True)

def plot_dataspace():
    dataset = "CIFAR10"
    data_space_plot(dataset=dataset, severity=1)

def plot_pred_dist():
    dataset = "CIFAR10"
    plot_prediction_distribution(architectures=['MediumCNN','WideResnet'], models=['MIMO','MIMBO'], M=3, dataset=dataset, severity=5, plot_baseline=True)

def visualise_kl_weighting():
    csv_path = r"reports\val_acc_BNN_kl_weightings.csv"
    kl_weighting_plot(csv_path)

def subnetwork_similarity(datasets, Ms, is_resnet, models=['MIMO', 'Naive', 'MIMBO']):
    for dataset in datasets:
        for M in Ms:
            try:
                checkpoint_list = []
                if is_resnet:
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMOWide/{dataset}/M{M}/C_MIMOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_NaiveWide/{dataset}/M{M}/C_NaiveWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBOWide/{dataset}/M{M}/C_MIMBOWide_28_10_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    architecture = 'Wide ResNet'
                else:
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMO/{dataset}/M{M}/C_MIMO_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_Naive/{dataset}/M{M}/C_Naive_{M}_members_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    checkpoint_list.append(torch.load(f'models/classification/checkpoints/C_MIMBO/{dataset}/M{M}/C_MIMBO_rep1_checkpoints.pt', map_location=torch.device('cpu')))
                    architecture = 'MediumCNN'
            except:
                print('Try again loser >:)')
            else:  
                for i, model in enumerate(models):
                    print(f"Computing disagreement and KL divergence for {model} model M = {M}")
                    print("Disagreement:", compute_disagreement(checkpoint_list[i][-1].argmax(dim=1))) # take last step in optimization trajectory, then argmax to get prediction
                    print("Average divergence", compute_KL_divergence(checkpoint_list[i][-1])) # take last step in optimization trajectory
                    print("\n")




if __name__ == '__main__':

    # accuracy_and_ECE()
    # function_space()
    # plot_example()
    # plot_pred_dist()
    function_space(Ms=[3], datasets=['CIFAR10'], is_resnet=False, use_axes=[1,2], twoD=True)
    # subnetwork_similarity(datasets=['CIFAR100'], Ms=[2,3,4,5], is_resnet=True, models=['MIMO', 'Naive', 'MIMBO'])

    
    # visualise_kl_weighting()


