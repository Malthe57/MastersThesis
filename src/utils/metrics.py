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
    
    # get one-hot encoded targets
    one_hot_correct_class = np.zeros(model_probabilities.shape, dtype=int) # shape: (N_test, C)
    #replacing 0 with a 1 at the index of the original array
    one_hot_correct_class[np.arange(correct_class.size),correct_class] = 1 # shape: (N_test, C)

    # loop over classes
    for i in range(model_probabilities.shape[1]):
        brier_score += np.power(model_probabilities[:,i]-one_hot_correct_class[:,i],2)

    # average over classes
    brier_score = brier_score / model_probabilities.shape[1]

    return np.mean(brier_score, axis=0) # average over test points

def compute_NLL(log_probs, targets):
    """
    This function computes the negative log likelihood of a model.
    log_probs is assumed to be in the shape (N_test, c), where C is the amount of classes, while N_test is the number of test points
    The targets input contains information on the ground truth class (N_test)
    """
    loss_fn = torch.nn.NLLLoss(reduction='mean')
    NLL = loss_fn(torch.tensor(log_probs), torch.tensor(targets))

    return NLL.item()

def compute_ECE(correct_predictions, confidence):
    reps = 1
    confidence = confidence[None, :]
    correct_predictions = correct_predictions[None, :]
    linspace = np.arange(0, 1.1, 0.1)
    bins_range = np.quantile(confidence.flatten(), linspace)
    n_samples = len(correct_predictions.T)
    
    conf_step_height = np.zeros((reps, 10))
    acc_step_height = np.zeros((reps,10))

    lengths = np.zeros((reps, 10))
    ECEs = np.zeros((reps, 10))
    for j in range(reps):
        for i in range(10):
            loc = np.where(np.logical_and(confidence[j,:]>=bins_range[i], confidence[j,:]<bins_range[i+1]))[0]
            if correct_predictions[j,loc].shape[0] != 0:
                acc_step_height[j, i] = np.mean(correct_predictions[j, loc])
                conf_step_height[j, i] = np.mean(confidence[j, loc])
                lengths[j, i] = correct_predictions[j,loc].shape[0]
                ECEs[j,i] = np.abs(acc_step_height[j, i]-conf_step_height[j, i])*lengths[j,i]
    
    ECE = np.sum(ECEs)/n_samples

    return ECE


if __name__ == "__main__":
    model_probabilities = np.array([[0.78, 0.08, 0.14], [0.05, 0.81, 0.14]])
    correct_class = np.array([0,1])

    print(compute_brier_score(model_probabilities, correct_class))