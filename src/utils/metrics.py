import torch
import numpy as np

def compute_regression_statistics(mu, sigma, y_test):
    mse = np.power(mu - y_test, 2)
    expected_mse = np.mean(mse, axis=1)
    andersen_fu_divergence = np.abs(np.mean(mse - np.power(sigma, 2), axis=1))

    return expected_mse, andersen_fu_divergence


def compute_brier_score(model_probabilities, correct_class):
    """
    This function computes the brier score of a model.
    model_probabilities is assumed to be in the shape (N_test, c), where C is the amount of classes, while N_test is the number of test points
    The correct_class input contains information on which class was correct for each test point.
    """
    brier_score = 0
    one_hot_correct_class = np.zeros((correct_class.size, correct_class.max()+1), dtype=int)
    #replacing 0 with a 1 at the index of the original array
    one_hot_correct_class[np.arange(correct_class.size),correct_class] = 1 

    for i in range(model_probabilities.shape[1]):
        brier_score += np.power(model_probabilities[:,i]-one_hot_correct_class[:,i],2)
    return np.mean(brier_score, axis=0)
