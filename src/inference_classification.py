import torch
from torch.utils.data import DataLoader 
import numpy as np
import argparse
from data.CIFAR10 import load_cifar10, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.CIFAR100 import load_cifar100
import glob
import os
from utils.metrics import compute_brier_score
from tqdm import tqdm
import time

def C_inference(model, testloader, device='cpu'):
    preds = []
    probs = []
    correct_preds = []
    targets = []

    for test_x, test_y in tqdm(testloader):
        test_x, test_y = test_x.float().to(device), test_y.type(torch.LongTensor).to(device)
        with torch.no_grad():
            log_probs, output, individual_outputs = model(test_x)

            preds.extend(output.cpu().detach().numpy())
            prob = np.exp(log_probs.cpu().detach().numpy())
            probs.extend(list(np.mean(prob, axis=0)))
            correct_preds.extend(list(output.cpu().detach().numpy()==test_y[:,0].cpu().detach().numpy()))
            targets.extend(list(test_y[:,0].cpu().detach().numpy()))   

    return np.array(preds), np.array(probs), np.array(correct_preds), np.array(targets)

def C_BNN_inference(model, testloader, device, n_classes=10):

    preds = []
    probs = []
    correct_preds = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, log_probs = model.inference(x_test, sample = True, n_samples=10, n_classes=n_classes)
            preds.extend(pred)
            probs.extend(np.exp(log_probs))
            correct_preds.extend(pred==y_test.cpu().detach().numpy())
            targets.extend(y_test.cpu().detach().numpy())

    return np.array(preds), np.array(probs), np.array(correct_preds), np.array(targets)

def C_MIMBO_inference(model, testloader, device, n_classes=10):
    preds = []
    probs = []
    correct_preds = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, mean_subnetwork_probs, mean_probs = model.inference(x_test, sample = True, n_samples=10, n_classes=n_classes)
            preds.extend(pred)
            probs.extend(np.exp(mean_probs))
            correct_preds.extend(pred==y_test[:,0].cpu().detach().numpy())
            targets.extend(y_test[:,0].cpu().detach().numpy())

    return np.array(preds), np.array(probs), np.array(correct_preds), np.array(targets)

def get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device= torch.device('cpu'), n_classes=10, reps=5):
    
    predictions_matrix = np.zeros((reps, len(model_paths), N_test))
    top_confidences_matrix = np.zeros((reps, len(model_paths), N_test))
    full_confidences_matrix = np.zeros((reps, len(model_paths), N_test, n_classes))
    correct_preds_matrix = np.zeros((reps, len(model_paths), N_test))
    brier_scores = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, len(model_paths), N_test))


    for i, paths in enumerate(model_paths):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        brier_scores_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, correct_preds, targets = C_inference(model, testloader, device)
        
            predictions_matrix[j, i, :] = preds
            top_confidences_matrix[j, i, :] = np.max(probs, axis=1)
            full_confidences_matrix[j, i,:,:] = probs
            correct_preds_matrix[j, i, :] = correct_preds
            targets_matrix[j, i, :] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))
        
        print(f"C_MIMO_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps, ddof=1)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
                
    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores

def get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device = torch.device('cpu'), n_classes=10, reps=5):

    predictions_matrix = np.zeros((reps, len(model_paths), N_test))
    top_confidences_matrix = np.zeros((reps, len(model_paths), N_test))
    full_confidences_matrix = np.zeros((reps, len(model_paths), N_test, n_classes))
    correct_preds_matrix = np.zeros((reps, len(model_paths), N_test))
    brier_scores = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, len(model_paths), N_test))

    for i, paths in enumerate(model_paths):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)
        
        brier_scores_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, correct_preds, targets = C_inference(model, testloader, device=device)

            predictions_matrix[j, i, :] = preds
            top_confidences_matrix[j, i, :] = np.max(probs, axis=1)
            full_confidences_matrix[j, i, :, :] = probs
            correct_preds_matrix[j, i, :] = correct_preds
            targets_matrix[j, i, :] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))

        print(f"C_Naive_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps, ddof=1)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
            
    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores

def get_C_bayesian_predictions(model_paths, testdata, batch_size, device = torch.device('cpu'), n_classes=10, reps=5):

    predictions_matrix = np.zeros((reps, 10000))
    top_confidences_matrix = np.zeros((reps, 10000))
    full_confidences_matrix = np.zeros((reps, 10000, n_classes))
    correct_preds_matrix = np.zeros((reps, 10000))
    brier_scores = np.zeros((reps))
    targets_matrix = np.zeros((reps, 10000))

    for i, model_path in enumerate(model_paths):
        model = torch.load(model_path, map_location=device)

        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True)
        preds, probs, correct_preds, targets = C_BNN_inference(model, testloader, device, n_classes=n_classes)
        
        targets_matrix[i, :] = targets
        brier_score = compute_brier_score(probs, targets)
        
        predictions_matrix[i, :] = preds
        top_confidences_matrix[i, :] = np.max(probs, axis=1)
        full_confidences_matrix[i, :, :] = probs
        correct_preds_matrix[i, :] = correct_preds
        brier_scores[i] = brier_score
    
    print(f"C_BNN Brier score: {np.mean(brier_scores)} \pm {1.96*np.std(brier_scores, ddof=1)}")


    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores

def get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device = torch.device('cpu'), n_classes=10, reps=5):
    predictions_matrix = np.zeros((reps, len(model_paths), N_test))
    top_confidences_matrix = np.zeros((reps, len(model_paths), N_test))
    full_confidences_matrix = np.zeros((reps, len(model_paths), N_test, n_classes))
    correct_preds_matrix = np.zeros((reps, len(model_paths), N_test))
    brier_scores = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, len(model_paths), N_test))

    for i, paths in enumerate(model_paths):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        brier_scores_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, correct_preds, targets = C_MIMBO_inference(model, testloader, device, n_classes=n_classes)

            predictions_matrix[j, i, :] = preds
            top_confidences_matrix[j, i, :] = np.max(probs, axis=1)
            full_confidences_matrix[j, i, :, :] = probs
            correct_preds_matrix[j, i, :] = correct_preds
            targets_matrix[j, i, :] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))

        print(f"C_MIMBO_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps, ddof=1)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
            
    return top_confidences_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores

def main(model_name, model_paths, Ms, dataset, n_classes, reps):
    _, _, testdata = load_cifar10("data/")
    batch_size = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'reports/Logs/{model_name}/{dataset}', exist_ok=True)
    print(f"Inference on {model_name} using {device}")

    match model_name:
        case "C_Baseline":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_mimo_predictions(model_paths, [1], testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMO/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_MIMO":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMO/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_MIMOWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMOWIde/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_Naive":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_Naive/{dataset}/{model_name}', predictions = predictions_matrix,confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_NaiveWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_NaiveWide/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_BNN":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_bayesian_predictions(model_paths, testdata, batch_size, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_BNN/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_BNNWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_bayesian_predictions(model_paths, testdata, batch_size, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_BNNWide/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_MIMBO":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, device=device, N_test=10000, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMBO/{dataset}/{model_name}',  predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
        case "C_MIMBOWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score = get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, device=device, N_test=10000, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMBOWide/{dataset}/{model_name}',  predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='C_MIMO', help='Model name [C_Baseline, C_MIMO, C_Naive, C_BNN, C_MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--resnet', action='store_true', default=False, help='Resnet model or not')
    
    args = parser.parse_args() 

    Ms = [int(M) for M in args.Ms[0].split(',')]
    reps = args.reps

    model_name = args.model_name
    if args.resnet:
        model_name += 'Wide'

    n_classes = args.n_classes
    if n_classes == 10:
        dataset = 'CIFAR10'
    else:
        dataset = 'CIFAR100'

    base_path = f'models/classification/{model_name}/{dataset}/'
    
    if args.model_name == "C_MIMO" or args.model_name == "C_Naive" or args.model_name == "C_MIMBO":
        M_path = [os.path.join(base_path, f"M{M}") for M in Ms]
        model_paths = [[os.path.join(p, model) for model in os.listdir(p)] for p in M_path]
    else:
        model_paths = [os.path.join(base_path, model) for model in os.listdir(base_path)]
   
    main(model_name, model_paths, Ms, dataset, n_classes, reps)
    print('done')