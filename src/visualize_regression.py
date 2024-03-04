import matplotlib.pyplot as plt
from matplotlib.patches import StepPatch
import torch
import numpy as np
import os
import pandas as pd
from visualization.visualize import plot_loss, plot_weight_distribution, plot_regression, reliability_diagram_regression
from data.OneD_dataset import ToyDataset
from utils.utils import get_training_min_max

# def unnormalise_outputs(mu, sigma):
#     x_train_min, x_train_max, y_train_min, y_train_max = get_training_min_max()
#     mu_unnormalised = (y_train_max - y_train_min) * (mu + 1) / 2 + y_train_min
#     sigma_unnormalised = sigma
#     return mu_unnormalised, sigma

if __name__ == '__main__':
    df_train = pd.read_csv("data/toydata/train_data.csv")
    df_test = pd.read_csv("data/toydata/test_data.csv")
    x_train, y_train = np.array(list(df_train['x'])), np.array(list(df_train['y']))
    x_test, y_test, line = np.array(list(df_test['x'])), np.array(list(df_test['y'])), np.array(list(df_test['line']))
    testdata = ToyDataset(x_test, y_test, normalise=True)

    MIMO = np.load("reports/Logs/MIMO/MIMO.npz")
    Naive = np.load("reports/Logs/Naive/Naive.npz")
    BNN = np.load("reports/Logs/BNN/BNN.npz")

    mimo_mu, mimo_mu_individual, mimo_sigma, mimo_sigma_individual = MIMO['predictions'], MIMO['mu_individual'], MIMO['predicted_std'], MIMO['sigma_individual']
    naive_mu, naive_mu_individual, naive_sigma, naive_sigma_individual = Naive['predictions'], Naive['mu_individual'], Naive['predicted_std'], Naive['sigma_individual']
    bnn_mu, bnn_sigma = BNN['predictions'], BNN['predicted_std']

    # unnormalise x_test and y_test
    # x_train_min, x_train_max, y_train_min, y_train_max = get_training_min_max()

    # mimo_mu, mimo_sigma = unnormalise_outputs(mimo_mu, mimo_sigma)
    # naive_mu, naive_sigma = unnormalise_outputs(naive_mu, naive_sigma)
    # bnn_mu, bnn_sigma = unnormalise_outputs(bnn_mu, bnn_sigma)

    Ms_mimo = list(range(2,2+naive_mu.shape[0]))
    Ms_naive = list(range(2,2+naive_mu.shape[0]))
    Ms_bnn = [bnn_mu.shape[0]]

    # plot regression
    plot_regression(x_train, y_train, x_test, y_test, line, mimo_mu, mimo_sigma, Ms=Ms_mimo, model_name="MIMO")
    plot_regression(x_train, y_train, x_test, y_test, line, naive_mu, naive_sigma, Ms=Ms_naive, model_name="Naive")
    plot_regression(x_train, y_train, x_test, y_test, line, bnn_mu, bnn_sigma, Ms=Ms_bnn, model_name="BNN")







