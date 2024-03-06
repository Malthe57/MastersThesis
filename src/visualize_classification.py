import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_single

if __name__ == '__main__':
    try:
        Baselines = np.load("reports/Logs/C_MIMO/C_Baseline.npz")
    except:
        print("No Baseline model found!")
        Baseline = False
    else:
        Baseline = True
        Predictions_Baseline, pred_individual_list_Baseline, confidences_matrix_Baseline, correct_preds_matrix_Baseline = Baselines["predictions"], Baselines["pred_individual"], Baselines["confidences"], Baselines["correct_preds"]

    try:
        MIMOs = np.load("reports/Logs/C_MIMO/C_MIMO.npz")
    except:
        print("No MIMO models found!")
        MIMO = False
    else:
        MIMO = True
        Predictions_MIMO, pred_individual_list_MIMO, confidences_matrix_MIMO, correct_preds_matrix_MIMO = MIMOs["predictions"], MIMOs["pred_individual"], MIMOs["confidences"], MIMOs["correct_preds"]
    
    try:
        Naives = np.load("reports/Logs/C_Naive/C_Naive.npz")
    except:
        print("No naive models found!")
        Naive = False
    else:
        Naive = True
        Predictions_Naive, pred_individual_list_Naive, confidences_matrix_Naive, correct_preds_matrix_Naive = Naives["predictions"], Naives["pred_individual"], Naives["confidences"], Naives["correct_preds"]
  
    try:
        BNNs = np.load("reports/Logs/C_BNN/C_BNN.npz")
    except:
        print("No bayesian model found!")
        BNN = False
    else:
        BNN = True
        Predictions_BNN, probabilities_BNN, correct_predictions_BNN, accuracy_BNN = BNNs["predictions"], BNNs["probabilities"], BNNs["correct_predictions"], BNNs["accuracy"]

    if Baseline:
        reliability_plot_classification_single(correct_predictions=correct_preds_matrix_Baseline[0,:], confidence=confidences_matrix_Baseline[0,:], model_name="C_Baseline")
        Baseline_accuracy = np.mean(correct_preds_matrix_Baseline, axis=1)
        print(f"Baseline test accuracy: {Baseline_accuracy[0]}\n")

    # if MIMO and Naive:
    #     MIMO_accuracies = np.mean(correct_preds_matrix_MIMO, axis=1)
    #     Naive_accuracies = np.mean(correct_preds_matrix_Naive, axis=1)
    #     for i in range(Predictions_MIMO.shape[0]):
    #         try:
    #             reliability_plot_classification_single(correct_predictions=correct_preds_matrix_MIMO[i,:], confidence = confidences_matrix_MIMO[i,:], naive_correct_predictions=correct_preds_matrix_Naive[i,:], naive_confidence=confidences_matrix_Naive[i,:], model_name=f"C_MIMO_M{i+2}", naive_model_name=f"C_Naive_M{i+2}", M=i+2)
    #         except:
    #             print(f"failed to plot reliability diagram for M{i+2}")
    #         else:
    #             print(f"MIMO M{i+2} test accuracy: {MIMO_accuracies[i]}")
    #             print(f"Naive M{i+2} test accuracy: {Naive_accuracies[i]}")
        
    if MIMO:
        MIMO_accuracies = np.mean(correct_preds_matrix_MIMO, axis=1)
        for i in range(Predictions_MIMO.shape[0]):
            reliability_plot_classification_single(correct_predictions=correct_preds_matrix_MIMO[i,:], confidence=confidences_matrix_MIMO[i,:], model_name="C_MIMO", M=i+2)
            print(f"C_MIMO M{i+2} test accuracy: {MIMO_accuracies[i]}\n")
    
    if Naive:
        Naive_accuracies = np.mean(correct_preds_matrix_Naive, axis=1)
        for i in range(Predictions_Naive.shape[0]):
            reliability_plot_classification_single(correct_predictions=correct_preds_matrix_Naive[i,:], confidence=confidences_matrix_Naive[i,:], model_name="C_Naive", M=i+2)
            print(f"C_Naive M{i+2} test accuracy: {Naive_accuracies[i]}\n")
            
                
    if BNN:
        reliability_plot_classification_single(correct_predictions=correct_predictions_BNN, confidence=probabilities_BNN, model_name="C_BNN")
        print(f"BNN test accuracy: {accuracy_BNN}\n")
    
