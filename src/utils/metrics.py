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
    one_hot_correct_class = np.zeros(model_probabilities.shape, dtype=int) # shape: (N_test, C)
    #replacing 0 with a 1 at the index of the original array
    one_hot_correct_class[np.arange(correct_class.size),correct_class] = 1 # shape: (N_test, C)

    for i in range(model_probabilities.shape[1]):
        brier_score += np.power(model_probabilities[:,i]-one_hot_correct_class[:,i],2)

    brier_score = brier_score / model_probabilities.shape[1]

    return np.mean(brier_score, axis=0)

def compute_NLL(log_probs, targets):
    """
    This function computes the negative log likelihood of a model.
    log_probs is assumed to be in the shape (N_test, c), where C is the amount of classes, while N_test is the number of test points
    The targets input contains information on the ground truth class (N_test)
    """
    loss_fn = torch.nn.NLLLoss()
    NLL = loss_fn(torch.tensor(log_probs), torch.tensor(targets))

    return NLL.item()


if __name__ == "__main__":
    model_probabilities = np.array([[0.78, 0.08, 0.14], [0.05, 0.81, 0.14]])
    correct_class = np.array([0,1])

    print(compute_brier_score(model_probabilities, correct_class))