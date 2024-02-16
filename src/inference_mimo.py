import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data import test_collate_fn, naive_collate_fn


def inference(model, testloader):
    predictions = []
    pred_individual = []

    for test_x, test_y in testloader:
        output, individual_outputs = model(test_x.float())
        
        predictions.extend(list(output.detach().numpy()))
        pred_individual.extend(list(individual_outputs.detach().numpy()))

    return np.array(predictions), np.array(pred_individual)

# get predictions and individual predictions for MIMO and Naive models
def get_mimo_predictions(model_path, Ms, testdata, N_test=200):

    predictions_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        predictions, pred_individual = inference(model, testloader)

        predictions_matrix[i, :] = predictions
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list

def get_naive_predictions(model_path, Ms, testdata, N_test=200):

    predictions_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: naive_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        predictions, pred_individual = inference(model, testloader)

        predictions_matrix[i, :] = predictions
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list