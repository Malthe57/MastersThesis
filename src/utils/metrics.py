import torch
import numpy as np

def compute_regression_statistics(mu, sigma, y_test):
    mse = np.power(mu - y_test, 2)
    expected_mse = np.mean(mse, axis=1)
    andersen_fu_divergence = np.abs(np.mean(mse - np.power(sigma, 2), axis=1))

    return expected_mse, andersen_fu_divergence


