import torch
from torch.utils.data import DataLoader 
import numpy as np
import argparse
from data.CIFAR10 import load_cifar10, load_CIFAR10C, C_train_collate_fn, C_test_collate_fn, C_Naive_train_collate_fn, C_Naive_test_collate_fn
from data.CIFAR100 import load_cifar100, load_CIFAR100C
import glob
import os
from utils.metrics import compute_brier_score, compute_NLL, compute_ECE
from utils.utils import logmeanexp
from tqdm import tqdm
import matplotlib.pyplot as plt

def sample_metrics(model_paths, testloader, device, n_classes=10, save_name='BNN'):
    accuracies = []
    acc_standard_errors = []
    brier_scores = []
    brier_standard_errors = []
    NLLs = []
    NLL_standard_errors = []
    ECEs = []
    ECE_standard_errors = []
    num_samples = [1,2,4,8,16,32]
    for i in num_samples:
        print("Number of samples:", i)
        rep_accuracies = [] # store accuracies for each rep
        rep_brier_scores = [] # store brier scores for each rep
        rep_NLLs = [] # store NLLs for each rep
        rep_ECEs = [] # store ECEs for each rep
        for model_path in model_paths:
            model = torch.load(model_path, map_location=device)
            if 'BNN' in save_name:
                preds, probs, log_probs, correct_preds, targets = C_BNN_inference(model, testloader, device, n_classes=n_classes, n_samples=i)
            else:
                preds, probs, log_probs, correct_preds, targets = C_MIMBO_inference(model, testloader, device, n_classes=n_classes, n_samples=i)
            rep_accuracies.append(np.mean(correct_preds)) # get accuracies for each rep
            rep_brier_scores.append(compute_brier_score(probs, targets))
            rep_NLLs.append(compute_NLL(log_probs, targets))
            rep_ECEs.append(compute_ECE(correct_preds, probs.max(axis=1))) # compute ECE using top probabilities

        accuracies.append(np.mean(rep_accuracies)) # compute mean of rep accuracies
        acc_standard_errors.append(np.std(rep_accuracies) / np.sqrt(len(rep_accuracies))) # compute standard error of the mean of rep accuracies
        brier_scores.append(np.mean(rep_brier_scores))
        brier_standard_errors.append(np.std(rep_brier_scores) / np.sqrt(len(rep_brier_scores)))
        NLLs.append(np.mean(rep_NLLs))
        NLL_standard_errors.append(np.std(rep_NLLs) / np.sqrt(len(rep_NLLs)))
        ECEs.append(np.mean(rep_ECEs))
        ECE_standard_errors.append(np.std(rep_ECEs) / np.sqrt(len(rep_ECEs)))
        
    np.savez(f"{save_name}_samples.npz", accuracies, acc_standard_errors, brier_scores, brier_standard_errors, NLLs, NLL_standard_errors, ECEs, ECE_standard_errors)
        
    # # plot 
    # plt.errorbar(np.array(num_samples), accuracies, yerr=acc_standard_errors, fmt='-o', ecolor='r', capsize=5)
    # plt.xlabel('Number of samples')
    # plt.ylabel('Accuracy')
    # plt.grid()
    # plt.title("Accuracy vs number of samples for BNN")
    # plt.tight_layout()
    # plt.savefig(f"reports/figures/acc_vs_samples_BNN.png", dpi=600)
    # plt.show()

    # plt.errorbar(np.array(num_samples), brier_scores, yerr=brier_standard_errors, fmt='-o', ecolor='r', capsize=5)
    # plt.xlabel('Number of samples')
    # plt.ylabel('Brier score')
    # plt.grid()
    # plt.title("Brier score vs number of samples for BNN")
    # plt.tight_layout()
    # plt.savefig(f"reports/figures/BS_vs_samples_BNN.png", dpi=600)
    # plt.show()

    # plt.errorbar(np.array(num_samples), NLLs, yerr=NLL_standard_errors, fmt='-o', ecolor='r', capsize=5)
    # plt.xlabel('Number of samples')
    # plt.ylabel('NLL')
    # plt.grid()
    # plt.title("NLL vs number of samples for BNN")
    # plt.tight_layout()
    # plt.savefig(f"reports/figures/NLL_vs_samples_BNN.png", dpi=600)
    # plt.show()

    # plt.errorbar(np.array(num_samples), ECEs, yerr=ECE_standard_errors, fmt='-o', ecolor='r', capsize=5)
    # plt.xlabel('Number of samples')
    # plt.ylabel('ECE')
    # plt.grid()
    # plt.title("ECE vs number of samples for BNN")
    # plt.tight_layout()
    # plt.savefig(f"reports/figures/ECE_vs_samples_BNN.png", dpi=600)
    # plt.show()

def C_inference(model, testloader, device='cpu'):
    preds = []
    probs = []
    log_probs = []
    correct_preds = []
    targets = []

    for test_x, test_y in tqdm(testloader):
        test_x, test_y = test_x.float().to(device), test_y.type(torch.LongTensor).to(device)
        with torch.no_grad():
            log_prob, output, individual_outputs = model(test_x)

            preds.extend(output.cpu().detach().numpy())
            prob = np.exp(logmeanexp(log_prob, dim=2).cpu().detach().numpy()) # convert from log-probs to probs
            log_probs.extend(list(logmeanexp(log_prob, dim=2).cpu().detach().numpy()))
            probs.extend(list(prob)) # take the mean over subnetworks dim
            correct_preds.extend(list(output.cpu().detach().numpy()==test_y[:,0].cpu().detach().numpy())) # get binary mask for predictions
            targets.extend(list(test_y[:,0].cpu().detach().numpy())) # get targets

    return np.array(preds), np.array(probs), np.array(log_probs), np.array(correct_preds), np.array(targets)

def C_BNN_inference(model, testloader, device, n_classes=10, n_samples=10):

    preds = []
    probs = []
    log_probs = []
    correct_preds = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, log_prob = model.inference(x_test, sample = True, n_samples=n_samples, n_classes=n_classes)
            preds.extend(pred)
            log_probs.extend(log_prob)
            probs.extend(np.exp(log_prob))
            correct_preds.extend(pred==y_test.cpu().detach().numpy())
            targets.extend(y_test.cpu().detach().numpy())

    return np.array(preds), np.array(probs), np.array(log_probs), np.array(correct_preds), np.array(targets)

def C_MIMBO_inference(model, testloader, device, n_classes=10, n_samples=10):
    preds = []
    probs = []
    log_probs = []
    correct_preds = []
    targets = []

    for x_test, y_test in tqdm(testloader):
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, mean_subnetwork_log_prob, mean_log_prob = model.inference(x_test, sample = True, n_samples=n_samples, n_classes=n_classes)
            preds.extend(pred)
            log_probs.extend(mean_log_prob)
            probs.extend(np.exp(mean_log_prob))
            correct_preds.extend(pred==y_test[:,0].cpu().detach().numpy())
            targets.extend(y_test[:,0].cpu().detach().numpy())

    return np.array(preds), np.array(probs), np.array(log_probs), np.array(correct_preds), np.array(targets)

def get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device= torch.device('cpu'), n_classes=10, reps=5):
    
    predictions_matrix = np.zeros((reps, N_test, len(model_paths)))
    top_confidences_matrix = np.zeros((reps, N_test, len(model_paths)))
    full_confidences_matrix = np.zeros((reps, N_test, n_classes, len(model_paths)))
    correct_preds_matrix = np.zeros((reps, N_test, len(model_paths)))
    brier_scores = np.zeros((reps, len(model_paths)))
    NLLs = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, N_test, len(model_paths)))


    for i, paths in enumerate(model_paths):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        brier_scores_reps = []
        NLL_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, log_probs, correct_preds, targets = C_inference(model, testloader, device)

            predictions_matrix[j, :, i] = preds
            top_confidences_matrix[j, :, i] = np.max(probs, axis=1)
            full_confidences_matrix[j,:,:, i] = probs
            correct_preds_matrix[j, :, i] = correct_preds
            targets_matrix[j, :, i] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))
            NLL_reps.append(compute_NLL(log_probs, targets))

        
        print(f"C_MIMO_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps)/np.sqrt(reps)}")
        print(f"C_MIMO_M{M} NLL: {np.mean(NLL_reps)} \pm {1.96*np.std(NLL_reps)/np.sqrt(reps)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
        NLLs[:, i] = np.array(NLL_reps)
                
    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores, NLLs

def get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device = torch.device('cpu'), n_classes=10, reps=5):

    predictions_matrix = np.zeros((reps, N_test, len(model_paths)))
    top_confidences_matrix = np.zeros((reps, N_test, len(model_paths)))
    full_confidences_matrix = np.zeros((reps, N_test, n_classes, len(model_paths)))
    correct_preds_matrix = np.zeros((reps, N_test, len(model_paths)))
    brier_scores = np.zeros((reps, len(model_paths)))
    NLLs = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, N_test, len(model_paths)))

    for i, paths in enumerate(model_paths):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_Naive_test_collate_fn(x, M), drop_last=False)
        
        brier_scores_reps = []
        NLL_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, log_probs, correct_preds, targets = C_inference(model, testloader, device=device)

            predictions_matrix[j, :, i] = preds
            top_confidences_matrix[j, :, i] = np.max(probs, axis=1)
            full_confidences_matrix[j, :, :, i] = probs
            correct_preds_matrix[j, :, i] = correct_preds
            targets_matrix[j, :, i] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))
            NLL_reps.append(compute_NLL(log_probs, targets))

        print(f"C_Naive_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps)/np.sqrt(reps)}")
        print(f"C_Naive_M{M} NLL: {np.mean(NLL_reps)} \pm {1.96*np.std(NLL_reps)/np.sqrt(reps)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
        NLLs[:, i] = np.array(NLL_reps)
            
    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores, NLLs

def get_C_bayesian_predictions(model_paths, testdata, batch_size, device = torch.device('cpu'), n_classes=10, reps=5):

    predictions_matrix = np.zeros((reps, 10000))
    top_confidences_matrix = np.zeros((reps, 10000))
    full_confidences_matrix = np.zeros((reps, 10000, n_classes))
    correct_preds_matrix = np.zeros((reps, 10000))
    brier_scores = np.zeros((reps))
    NLLs = np.zeros((reps))
    targets_matrix = np.zeros((reps, 10000))

    for i, model_path in enumerate(model_paths):
        model = torch.load(model_path, map_location=device)

        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True)
        preds, probs, log_probs, correct_preds, targets = C_BNN_inference(model, testloader, device, n_classes=n_classes)
        
        targets_matrix[i, :] = targets
        predictions_matrix[i, :] = preds
        top_confidences_matrix[i, :] = np.max(probs, axis=1)
        full_confidences_matrix[i, :, :] = probs
        correct_preds_matrix[i, :] = correct_preds
        brier_scores[i] = compute_brier_score(probs, targets)
        NLLs[i] = compute_NLL(log_probs, targets)
    
    print(f"C_BNN Brier score: {np.mean(brier_scores)} \pm {1.96*np.std(brier_scores)/np.sqrt(reps)}")
    print(f"C_BNN NLL: {np.mean(NLLs)} \pm {1.96*np.std(NLLs)/np.sqrt(reps)}")

    return predictions_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores, NLLs

def get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, N_test=200, device = torch.device('cpu'), n_classes=10, reps=5):
    predictions_matrix = np.zeros((reps, N_test, len(model_paths)))
    top_confidences_matrix = np.zeros((reps, N_test, len(model_paths)))
    full_confidences_matrix = np.zeros((reps, N_test, n_classes, len(model_paths)))
    correct_preds_matrix = np.zeros((reps, N_test, len(model_paths)))
    brier_scores = np.zeros((reps, len(model_paths)))
    NLLs = np.zeros((reps, len(model_paths)))
    targets_matrix = np.zeros((reps, N_test, len(model_paths)))

    for i, paths in enumerate(model_paths):
        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)

        brier_scores_reps = []
        NLL_reps = []
        for j, model_path in enumerate(paths):
            model = torch.load(model_path, map_location = device)
            preds, probs, log_probs, correct_preds, targets = C_MIMBO_inference(model, testloader, device, n_classes=n_classes)

            predictions_matrix[j, :, i] = preds
            top_confidences_matrix[j, :, i] = np.max(probs, axis=1)
            full_confidences_matrix[j, :, :, i] = probs
            correct_preds_matrix[j, :, i] = correct_preds
            targets_matrix[j, :, i] = targets
            brier_scores_reps.append(compute_brier_score(probs, targets))
            NLL_reps.append(compute_NLL(log_probs, targets))

        print(f"C_MIMBO_M{M} Brier score: {np.mean(brier_scores_reps)} \pm {1.96*np.std(brier_scores_reps)/np.sqrt(reps)}")
        print(f"C_MIMBO_M{M} NLL: {np.mean(NLL_reps)} \pm {1.96*np.std(NLL_reps)/np.sqrt(reps)}")
        brier_scores[:, i] = np.array(brier_scores_reps)
        NLLs[:, i] = np.array(NLL_reps)
            
    return top_confidences_matrix, top_confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_scores, NLLs

def main(model_name, model_paths, Ms, dataset, n_classes, reps, ood, severity):
    if ood:
        testdata = load_CIFAR10C("data/CIFAR-10-C/", "impulse_noise", severity=severity) if n_classes == 10 else load_CIFAR100C("data/CIFAR-100-C/", "impulse_noise", severity=severity)
        model_name += f"_severity{severity}"
    else:
        _, _, testdata = load_cifar10("data/") if n_classes == 10 else load_cifar100("data/")
    batch_size = 500
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'reports/Logs/{model_name}/{dataset}', exist_ok=True)
    print(f"Inference on {model_name} on dataset with {n_classes} classes using {device} ")

    match model_name:
        case "C_Baseline":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_mimo_predictions(model_paths, [1], testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMO/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_MIMO":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMO/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_MIMOWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_mimo_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMOWide/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_Naive":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_Naive/{dataset}/{model_name}', predictions = predictions_matrix,confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_NaiveWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_naive_predictions(model_paths, Ms, testdata, batch_size, N_test=10000, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_NaiveWide/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_BNN":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_bayesian_predictions(model_paths, testdata, batch_size, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_BNN/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_BNNWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_bayesian_predictions(model_paths, testdata, batch_size, device = device, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_BNNWide/{dataset}/{model_name}', predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_MIMBO":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, device=device, N_test=10000, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMBO/{dataset}/{model_name}',  predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
        case "C_MIMBOWide":
            predictions_matrix, confidences_matrix, full_confidences_matrix, correct_preds_matrix, targets_matrix, brier_score, NLL = get_C_mimbo_predictions(model_paths, Ms, testdata, batch_size, device=device, N_test=10000, n_classes=n_classes, reps=reps)
            np.savez(f'reports/Logs/C_MIMBOWide/{dataset}/{model_name}',  predictions = predictions_matrix, confidences = confidences_matrix, full_confidences = full_confidences_matrix, correct_preds = correct_preds_matrix, targets_matrix=targets_matrix, brier_score = brier_score, NLL=NLL)
            

if __name__ == "__main__":
    # investigate sampling efficiency
    sampling_efficiency = False

    parser = argparse.ArgumentParser(description='Inference for MIMO, Naive, and BNN models')
    parser.add_argument('--model_name', type=str, default='C_MIMO', help='Model name [C_Baseline, C_MIMO, C_Naive, C_BNN, C_MIBMO]')
    parser.add_argument('--Ms', nargs='+', default="2,3,4,5", help='Number of subnetworks for MIMO and Naive models')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--resnet', action='store_true', default=False, help='Resnet model or not')
    parser.add_argument('--ood', action='store_true', default=False, help='Use CIFAR10 corrupted data or not. Should always be False for CIFAR100')
    parser.add_argument('--severity', type=int, default=5, help='Severity of corruption')
    
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

    # base_path = f'models/classification/{model_name}/{dataset}/'
    base_path = f'models/Orig_resnet_cifar100/{model_name}/{dataset}/'
    
    if args.model_name == "C_MIMO" or args.model_name == "C_Naive" or args.model_name == "C_MIMBO":
        M_path = [os.path.join(base_path, f"M{M}") for M in Ms]
        model_paths = [[os.path.join(p, model) for model in os.listdir(p)[:reps]] for p in M_path]
    else:
        model_paths = [os.path.join(base_path, model) for model in os.listdir(base_path)]
    
    if args.ood:
        dataset += '_C'

    if sampling_efficiency:
        _, _, testdata = load_cifar10("data/") if n_classes == 10 else load_cifar100("data/")
        batch_size = 500

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.model_name == 'C_BNN':
            testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True)
            sample_metrics(model_paths=model_paths, testloader=testloader, device=device, n_classes=n_classes, save_name='BNN')

        elif args.model_name == 'C_MIMBO':
            testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=lambda x: C_test_collate_fn(x, M), drop_last=False)
            for i, M in enumerate(Ms):
                    sample_metrics(model_paths=model_paths[i], testloader=testloader, device=device, n_classes=n_classes, save_name=f'MIMBO{M}')

    else:
        main(model_name, model_paths, Ms, dataset, n_classes, reps, args.ood, args.severity)
        print('done')

