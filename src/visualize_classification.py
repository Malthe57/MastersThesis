import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification, reliability_plot_classification_bayesian

if __name__ == '__main__':
    try:
        MIMOs = np.load("reports/Logs/MIMO/C_MIMO.npz")
    except:
        print("No MIMO models found!")
        MIMO = False
    else:
        MIMO = True
        Predictions_MIMO, pred_individual_list_MIMO, confidences_matrix_MIMO, correct_preds_matrix_MIMO = MIMOs["predictions"], MIMOs["pred_individual"], MIMOs["confidences"], MIMOs["correct_preds"]
    
    try:
        Naives = np.load("reports/Logs/Naive/C_Naive.npz")
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

    if MIMO and Naive:
        for i in range(Predictions_MIMO.shape[0]):
            reliability_plot_classification(correct_predictions=correct_preds_matrix_MIMO[:,i], confidence = confidences_matrix_MIMO[:,i], naive_correct_predictions=correct_preds_matrix_Naive[:,i], naive_confidence=confidences_matrix_Naive[:,i], model_name=f"C_MIMO_M{i}", naive_model_name=f"C_Naive_M{i}", M=i)

    if BNN:
        reliability_plot_classification_bayesian(correct_predictions=correct_predictions_BNN, confidence=probabilities_BNN, model_name="BNNs")

    
