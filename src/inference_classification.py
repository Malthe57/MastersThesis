import torch
from torch.utils.data import DataLoader 
import numpy as np
import argparse
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
import glob
import os

def C_inference(model, testloader):
    predictions = []
    pred_individual = []
    confidences = []
    conf_individual = []

    for test_x, test_y in testloader:

        log_prob, output, individual_outputs = model(test_x)

        predictions.extend(output.detach().numpy())
        pred_individual.extend(list(individual_outputs.detach().numpy()))

        prob = np.exp(log_prob.detach().numpy())
        confidences.extend(list(np.mean(prob,axis=0)))
        conf_individual.extend(list(prob))

        correct_preds = output==test_y[:,0]

    return np.array(predictions), np.array(pred_individual), np.array(confidences), np.array(conf_individual), np.array(correct_preds)

def C_BNN_inference(model, testloader, device):

    preds = []
    log_probs = []
    targets = []

    for x_test, y_test in testloader:
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, probs = model.inference(x_test, sample = True, n_samples=10, n_classes= 10)
            preds.extend(pred)
            log_probs.extend(probs)
            targets.extend(y_test.cpu().detach().numpy())

    return np.array(preds), np.array(log_probs), np.array(targets)

def get_C_mimo_predictions(model_path, Ms, testdata, N_test=200, device= torch.device('cpu')):

    predictions_matrix = np.zeros((len(model_path), N_test))
    confidences_matrix = np.zeros((len(model_path), N_test))
    correct_preds_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):
        print(model_path)
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        model = torch.load(model, map_location = device)
        predictions, pred_individual, confidences, conf_individual, correct_preds = C_inference(model, testloader)

        predictions_matrix[i, :] = predictions
        confidences_matrix[i, :] = np.max(confidences, axis=1)
        correct_preds_matrix[i, :] = correct_preds
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix

def get_C_naive_predictions(model_path, Ms, testdata, N_test=200, device = torch.device('cpu')):

    predictions_matrix = np.zeros((len(model_path), N_test))
    confidences_matrix = np.zeros((len(model_path), N_test))
    correct_preds_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)

        model = torch.load(model, map_location = device)
        predictions, pred_individual, confidences, conf_individual, correct_preds = C_inference(model, testloader)

        predictions_matrix[i, :] = predictions
        confidences_matrix[i, :] = np.max(confidences,axis=1)
        correct_preds_matrix[i, :] = correct_preds
        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix

def get_C_bayesian_predictions(model_path, testdata, batch_size, device = torch.device('cpu')):
    model = torch.load(model_path[0], map_location=device)

    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True)
    predictions, log_probabilities, targets = C_BNN_inference(model, testloader, device)
    
    probs = np.exp(log_probabilities)
    correct_predictions = predictions==targets
    accuracy = np.sum(correct_predictions)/len(correct_predictions)

    return predictions, probs, correct_predictions, accuracy

def main(model_name, model_path, Ms):
    _, _, testdata = load_cifar("data/")
    batch_size = 500

    match model_name:
        case "C_Baseline":
            predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix = get_C_mimo_predictions(model_path, [1], testdata, N_test=10000)
            np.savez(f'reports/Logs/MIMO/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, correct_preds = correct_preds_matrix)
        case "C_MIMO":
            predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix = get_C_mimo_predictions(model_path, Ms, testdata, N_test=10000)
            np.savez(f'reports/Logs/MIMO/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, correct_preds = correct_preds_matrix)
        case "C_Naive":
            predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix = get_C_naive_predictions(model_path, Ms, testdata, N_test=10000)
            np.savez(f'reports/Logs/Naive/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, correct_preds = correct_preds_matrix)
        case "C_BNN":
            predictions, probabilities, correct_predictions, accuracy = get_C_bayesian_predictions(model_path, testdata, batch_size)
            np.savez(f'reports/Logs/BNN/{model_name}', predictions = predictions, probabilities = probabilities, correct_predictions = correct_predictions, accuracy = accuracy)
        case "C_MIBMO":
            pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='MIMO', help='Model name [Baseline, MIMO, Naive, BNN, MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="1,2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    args = parser.parse_args()

    base_path = f'models/classification/{args.model_name}'
    model_path = [model for model in glob.glob(os.path.join(base_path,'*.pt'))]
    print(args.Ms)
    Ms = [int(M) for M in args.Ms[0].split(',')]
    main(args.model_name, model_path, Ms)
    print('done')