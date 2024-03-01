import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification

if __name__ == '__main__':
    try:
        MIMOs = np.load("reports/Logs/MIMO/MIMO.npz")
        MIMO = True
        Predictions_MIMO, pred_individual_list_MIMO, confidences_matrix_MIMO, correct_preds_matrix_MIMO = MIMOs["predictions"], MIMOs["pred_individual"], MIMOs["confidences"], MIMOs["correct_preds"]
    except:
        MIMO = False
    
    try:
        Naives = np.load("reports/Logs/Naive/Naive.npz")
        Naive = True
        Predictions_Naive, pred_individual_list_Naive, confidences_matrix_Naive, correct_preds_matrix_Naive = Naives["predictions"], Naives["pred_individual"], Naives["confidences"], Naives["correct_preds"]
    except:
        Naive = False
    
    try:
        BNNs = np.load("report/Logs/BNN/BNN.npz")
        Predictions_BNN, probabilities_BNN, correct_predictions_BNN, accuracy_BNN = BNNs["predictions"], BNNs["probabilities"], BNNs["correct_predictions"], BNNs["accuracy"]

    if MIMO:
        for i in range(Predictions_MIMO.shape()[1]):
            reliability_plot_classification(correct_predictions=correct_preds_matrix_MIMO[], confidence = confidences, naive_correct_predictions=Naive_correct_predictions, naive_confidence=Naive_confidences, model_name=model_name, naive_model_name=naive_model_name, M=M)

    
