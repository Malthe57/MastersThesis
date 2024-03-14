import torch
from torch.utils.data import DataLoader 
import numpy as np
import argparse
from data.CIFAR10 import load_cifar, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
import glob
import os
from utils.metrics import compute_brier_score
from tqdm import tqdm

def C_inference(model, testloader, device='cpu'):
    predictions = []
    pred_individual = []
    confidences = []
    conf_individual = []
    correct_preds = []
    test_ys = []

    for test_x, test_y in tqdm(testloader):

        test_x, test_y = test_x.float().to(device), test_y.type(torch.LongTensor).to(device)

        log_prob, output, individual_outputs = model(test_x)

        predictions.extend(output.detach().numpy())
        pred_individual.extend(list(individual_outputs.detach().numpy()))

        prob = np.exp(log_prob.detach().numpy())
        confidences.extend(list(np.mean(prob, axis=0)))
        conf_individual.extend(list(prob))

        correct_preds.extend(list(output==test_y[:,0]))
        test_ys.extend(list(test_y[:,0]))   



    return np.array(predictions), np.array(pred_individual), np.array(confidences), np.array(conf_individual), np.array(correct_preds), np.array(test_ys)

def C_BNN_inference(model, testloader, device):

    preds = []
    log_probs = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, probs = model.inference(x_test, sample = True, n_samples=10, n_classes= 10)
            preds.extend(pred)
            log_probs.extend(probs)
            targets.extend(y_test.cpu().detach().numpy())

    return np.array(preds), np.array(log_probs), np.array(targets)

def C_MIMBO_inference(model, testloader, device):
    preds = []
    log_probs = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, mean_subnetwork_probs, mean_probs = model.inference(x_test, sample = True, n_samples=10, n_classes= 10)
            preds.extend(pred)
            log_probs.extend(mean_probs)
            targets.extend(y_test.cpu().detach().numpy())

    return np.array(preds), np.array(log_probs), np.array(targets)

def get_C_mimo_predictions(model_path, Ms, testdata, N_test=200, device= torch.device('cpu')):

    predictions_matrix = np.zeros((len(model_path), N_test))
    confidences_matrix = np.zeros((len(model_path), N_test))
    full_confidences_matrix = np.zeros((len(model_path), N_test, 10))
    correct_preds_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=500, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        model = torch.load(model, map_location = device)
        predictions, pred_individual, confidences, conf_individual, correct_preds, targets = C_inference(model, testloader, device)

        predictions_matrix[i, :] = predictions
        confidences_matrix[i, :] = np.max(confidences, axis=1)
        full_confidences_matrix[i,:,:] = confidences
        correct_preds_matrix[i, :] = correct_preds
        pred_individual_list.append(pred_individual)
        brier_score = compute_brier_score(confidences, targets)
        print(f"C_MIMO_M{M} Brier score: {brier_score}")
            
    return predictions_matrix, np.concatenate(pred_individual_list, axis=1), confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score

def get_C_naive_predictions(model_path, Ms, testdata, N_test=200, device = torch.device('cpu')):

    predictions_matrix = np.zeros((len(model_path), N_test))
    confidences_matrix = np.zeros((len(model_path), N_test))
    full_confidences_matrix = np.zeros((len(model_path), N_test, 10))
    correct_preds_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=500, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)

        model = torch.load(model, map_location = device)
        predictions, pred_individual, confidences, conf_individual, correct_preds, targets = C_inference(model, testloader)

        predictions_matrix[i, :] = predictions
        confidences_matrix[i, :] = np.max(confidences,axis=1)
        full_confidences_matrix[i, :, :] = confidences
        correct_preds_matrix[i, :] = correct_preds
        brier_score = compute_brier_score(confidences, targets)
        print(f"C_Naive_M{M} Brier score: {brier_score}")

        pred_individual_list.append(pred_individual)
            
    return predictions_matrix, np.concatenate(pred_individual_list, axis=1), confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score

def get_C_bayesian_predictions(model_path, testdata, batch_size, device = torch.device('cpu')):
    model = torch.load(model_path[0], map_location=device)

    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True)
    predictions, log_probabilities, targets = C_BNN_inference(model, testloader, device)
    
    probs = np.exp(log_probabilities)
    correct_predictions = predictions==targets
    accuracy = np.sum(correct_predictions)/len(correct_predictions)
    top_probs = np.max(probs, axis=1)
    brier_score = compute_brier_score(probs, targets)
    print(f"C_BNN Brier score: {brier_score}")

    return predictions, probs, top_probs, correct_predictions, accuracy, brier_score

def get_C_mimbo_predictions(model_path, Ms, testdata, N_test=200, device = torch.device('cpu')):
    top_probs_matrix = np.zeros((len(model_path), N_test))
    correct_pred_matrix = np.zeros((len(model_path), N_test))

    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=500, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        model = torch.load(model, map_location = device)
        predictions, log_probs, targets = C_MIMBO_inference(model, testloader, device)

        probs = np.exp(log_probs)
        correct_predictions = predictions==targets[:,0]
        top_probs = np.max(probs, axis=1)
        brier_score = compute_brier_score(probs, targets[:,0])
        print(f"C_MIMBO_M{M} Brier score: {brier_score}")

        top_probs_matrix[i, :] = top_probs
        correct_pred_matrix[i, :] = correct_predictions

    return top_probs_matrix, correct_pred_matrix

def main(model_name, model_path, Ms):
    _, _, testdata = load_cifar("data/")
    batch_size = 500

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Inference using {device}")

    match model_name:
        case "C_Baseline":
            predictions_matrix, pred_individual_list, confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score = get_C_mimo_predictions(model_path, [1], testdata, N_test=10000, device = device)
            np.savez(f'reports/Logs/C_MIMO/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, brier_score = brier_score)
        case "C_MIMO":
            predictions_matrix, pred_individual_list, confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score = get_C_mimo_predictions(model_path, Ms, testdata, N_test=10000, device = device)
            np.savez(f'reports/Logs/C_MIMO/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, brier_score = brier_score)
        case "C_MIMOWide_28_10":
            predictions_matrix, pred_individual_list, confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score = get_C_mimo_predictions(model_path, Ms, testdata, N_test=10000, device = device)
            np.savez(f'reports/Logs/C_MIMO/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, brier_score = brier_score)
        case "C_Naive":
            predictions_matrix, pred_individual_list, confidences_matrix, full_confidences_matrix, correct_preds_matrix, brier_score = get_C_naive_predictions(model_path, Ms, testdata, N_test=10000, device = device)
            np.savez(f'reports/Logs/C_Naive/{model_name}', predictions = predictions_matrix, pred_individual = pred_individual_list, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, brier_score = brier_score)
        case "C_BNN":
            predictions, full_probabilities, probabilities, correct_predictions, accuracy, brier_score = get_C_bayesian_predictions(model_path, testdata, batch_size, device = device)
            np.savez(f'reports/Logs/C_BNN/{model_name}', predictions = predictions, probabilities = probabilities, full_probabilities = full_probabilities, correct_predictions = correct_predictions, accuracy = accuracy, brier_score = brier_score)
        case "C_MIMBO":
            top_probabilities, correct_predictions = get_C_mimbo_predictions(model_path, Ms, testdata, N_test=500)
            np.savez(f'reports/Logs/C_MIMBO/{model_name}', top_probabilities = top_probabilities, correct_predictions = correct_predictions)
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='C_MIMO', help='Model name [C_Baseline, C_MIMO, C_Naive, C_BNN, C_MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    parser.add_argument('--resnet', action='store_true', default=False, help='Resnet model or not')
    
    args = parser.parse_args()

    Ms = [int(M) for M in args.Ms[0].split(',')]

    model_name = args.model_name
    if args.resnet:
        model_name += 'Wide_28_10'

    base_path = f'models/classification/{model_name}'
    if args.model_name == "C_Baseline":
        base_path = 'models/classification/C_MIMO'
        model_path = [os.path.join(base_path, f"{model_name}.pt")]
    elif args.model_name == "C_MIMO" or model_name == "C_Naive":
        model_path = [model for model in [os.path.join(base_path,f'{model_name}_{M}_members.pt') for M in Ms]]
    elif args.model_name == "C_MIMBO":
        model_path = [model for model in [os.path.join(base_path,f'{model_name}_{M}_members.pt') for M in Ms]]
    else:
        model_path = [os.path.join(base_path, f"{model_name}.pt")]
   
    main(args.model_name, model_path, Ms)
    print('done')