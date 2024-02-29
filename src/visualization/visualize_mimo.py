import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression, reliability_plot_classification

if __name__ == '__main__':
    #init params
    modes = ['regression','classification']
    mode = modes[0]
    model_names = ['MIMO','Naive', 'VarMIMO', 'C_MIMO', 'C_Naive']
    model_name = model_names[2]
    naive_model_name = "C_Naive"
    base_load_path = 'reports/Logs/'
    load_path = os.path.join(base_load_path, model_name)
    naive_load_path = os.path.join(base_load_path, naive_model_name)
    # Ms = [2,3,4]
    Ms = [3]
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

    
