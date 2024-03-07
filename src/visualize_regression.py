import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression
from data.OneD_dataset import ToyDataset
from utils.utils import get_training_min_max
from utils.metrics import compute_regression_statistics

if __name__ == '__main__':
    df_train = pd.read_csv("data/toydata/train_data.csv")
    df_test = pd.read_csv("data/toydata/test_data.csv")
    x_train, y_train = np.array(list(df_train['x'])), np.array(list(df_train['y']))
    x_test, y_test, line = np.array(list(df_test['x'])), np.array(list(df_test['y'])), np.array(list(df_test['line']))
    testdata = ToyDataset(x_test, y_test, normalise=True)

    MIMO = np.load("reports/Logs/MIMO/MIMO.npz")
    Naive = np.load("reports/Logs/Naive/Naive.npz")
    BNN = np.load("reports/Logs/BNN/BNN.npz")
    MIMBO = np.load("reports/Logs/MIMBO/MIMBO.npz")

    mimo_mu, mimo_mu_individual, mimo_sigma, mimo_sigma_individual = MIMO['predictions'], MIMO['mu_individual'], MIMO['predicted_std'], MIMO['sigma_individual']
    naive_mu, naive_mu_individual, naive_sigma, naive_sigma_individual = Naive['predictions'], Naive['mu_individual'], Naive['predicted_std'], Naive['sigma_individual']
    bnn_mu, bnn_sigma = BNN['predictions'], BNN['predicted_std']
    mimbo_mu, mimbo_mu_individual, mimbo_sigma, mimbo_sigma_individual = MIMBO['predictions'], MIMBO['mu_individual'], MIMBO['predicted_std'], MIMBO['sigma_individual']

    Ms_mimo = list(range(2,2+naive_mu.shape[0]))
    Ms_naive = list(range(2,2+naive_mu.shape[0]))
    Ms_bnn = [bnn_mu.shape[0]]
    Ms_mimbo = [2]

    # plot variances
    plt.plot(x_test, mimo_sigma[0], label="Baseline")
    for i in range(len(Ms_mimo)):
        M = Ms_mimo[i]
        plt.plot(x_test, mimo_sigma[i+1], label=f"MIMO, M={M}")
    plt.legend()
    plt.show()

    for i in range(len(Ms_naive)):
        M = Ms_naive[i]
        plt.plot(x_test, naive_sigma[i], label=f"Naive, M={M}")
    plt.legend()
    plt.show()

    plt.plot(x_test, bnn_sigma[0], label="BNN")
    plt.legend()
    plt.show()

    for i in range(len(Ms_mimbo)):
        M = Ms_mimbo[i]
        plt.plot(x_test, mimbo_sigma[i], label=f"MIMBO, M={M}")
    plt.show()


    # plot regression
    plot_regression(x_train, y_train, x_test, y_test, line, mimo_mu, mimo_sigma, Ms=Ms_mimo, model_name="MIMO")
    plot_regression(x_train, y_train, x_test, y_test, line, naive_mu, naive_sigma, Ms=Ms_naive, model_name="Naive")
    plot_regression(x_train, y_train, x_test, y_test, line, bnn_mu, bnn_sigma, Ms=Ms_bnn, model_name="BNN")
    plot_regression(x_train, y_train, x_test, y_test, line, mimbo_mu, mimbo_sigma, Ms=Ms_mimbo, model_name="MIMBO")

    # plot reliability diagrams¨
    reliability_diagram_regression(mimo_mu[0], y_test, mimo_sigma[0], M=1, model_name="Baseline")
    for i in range(len(Ms_mimo)):
        M = Ms_mimo[i]
        reliability_diagram_regression(mimo_mu[i+1], y_test, mimo_sigma[i+1], M=M, model_name="MIMO")
    for i in range(len(Ms_naive)):
        M = Ms_naive[i]
        reliability_diagram_regression(naive_mu[i], y_test, naive_sigma[i], M=M, model_name="Naive")
        
    reliability_diagram_regression(bnn_mu, y_test, bnn_sigma, M=1, model_name="BNN")

    for i in range(len(Ms_mimbo)):
        M = Ms_mimbo[i]
        reliability_diagram_regression(mimbo_mu[i], y_test, mimbo_sigma[i], M=M, model_name="MIMBO")

    # compute statistics
    mmse, ece = compute_regression_statistics(mimo_mu, mimo_sigma, y_test)
    naive_mmse, naive_ece = compute_regression_statistics(naive_mu, naive_sigma, y_test)
    bnn_mmse, bnn_ece = compute_regression_statistics(bnn_mu, bnn_sigma, y_test)
    mimbo_mmse, mimbo_ece = compute_regression_statistics(mimbo_mu, mimbo_sigma, y_test)

    print(f"MIMO: MMSE: {mmse}, ECE: {ece}")
    print(f"Naive: MMSE: {naive_mmse}, ECE: {naive_ece}")
    print(f"BNN: MMSE: {bnn_mmse}, ECE: {bnn_ece}")
    print(f"MIMBO: MMSE: {mimbo_mmse}, ECE: {mimbo_ece}")






