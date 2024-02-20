import torch
import numpy as np
from torch.utils.data import DataLoader
from data.OneD_dataset import test_collate_fn, naive_collate_fn, generate_data, ToyDataset
from data.CIFAR10 import C_Naive_test_collate_fn, C_Naive_train_collate_fn, C_test_collate_fn, C_train_collate_fn, load_cifar
from visualization.visualize_mimo import reliability_plot, plot_regression
import glob
import os
from models.mimo import C_MIMONetwork, C_NaiveNetwork, MIMONetwork, NaiveNetwork



def inference(model, testloader):
    predictions = []
    pred_individual = []

    for test_x, test_y in testloader:
        output, individual_outputs = model(test_x.float())
        
        predictions.extend(list(output.detach().numpy()))
        pred_individual.extend(list(individual_outputs.detach().numpy()))

    return np.array(predictions), np.array(pred_individual)

def var_inference(model, testloader):
    mu_list = []
    sigma_list = []
    mus_list = []
    sigmas_list = []

    for test_x, test_y in testloader:
        mu, sigma, mus, sigmas = model(test_x.float())
        
        mu_list.extend(list(mu.detach().numpy()))
        sigma_list.extend(list(sigma.detach().numpy()))
        mus_list.extend(list(mus.detach().numpy()))
        sigmas_list.extend(list(sigmas.detach().numpy()))

    return np.array(mu_list), np.array(sigma_list), np.array(mus_list), np.array(sigmas_list)

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

def get_var_mimo_predictions(model_path, Ms, testdata, N_test=200):

    mu_matrix = np.zeros((len(model_path), N_test))
    sigma_matrix = np.zeros((len(model_path), N_test))
    mu_individual_list = []
    sigma_individual_list = []


    for i, model in enumerate(model_path):

        M = Ms[i]
        testloader = DataLoader(testdata, batch_size=N_test, shuffle=False, collate_fn=lambda x: test_collate_fn(x, M), drop_last=False)

        model = torch.load(model)
        mu, sigma, mus, sigmas = var_inference(model, testloader)

        mu_matrix[i, :] = mu
        sigma_matrix[i, :] = sigma
        mu_individual_list.append(mus)
        sigma_individual_list.append(sigmas)
            
    return mu_matrix, sigma_matrix, mu_individual_list, sigma_individual_list

def get_C_mimo_predictions(model_path, Ms, testdata, N_test=200, device= torch.device('cpu')):

    predictions_matrix = np.zeros((len(model_path), N_test))
    confidences_matrix = np.zeros((len(model_path), N_test))
    correct_preds_matrix = np.zeros((len(model_path), N_test))
    pred_individual_list = []

    for i, model in enumerate(model_path):

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

if __name__ == '__main__':
    model_names = ['MIMO','Naive','VarMIMO','C_MIMO','C_Naive']
    model_name = model_names[2]
    naive = False
    modes = ['Regression','Classification']
    mode = modes[0]

    # Ms = [2,3,4]
    Ms = [3]
    N_test = 500
    # base_path = "notebooks/C_MIMO_ensembles/"
    base_path = 'models/'
    model_path = [model for model in glob.glob(os.path.join(base_path,'*.pt'))]
    data_path = "data/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_var = True
    
    
    if mode == 'Regression':
        #prepare data
        #set parameters for toy data
        lower = -0.5
        upper = 1.5
        std = 0.02
        
        x_test, y_test = generate_data(N_test, lower, upper, std)
        testdata = ToyDataset(x_test, y_test)


        if naive == True:
            predictions_matrix, pred_individual_list = get_naive_predictions(model_path = model_path, Ms = Ms, N_test = N_test, testdata = testdata)
        else:
            if is_var:
                mu, sigma, mus, sigmas = get_var_mimo_predictions(model_path=model_path, Ms = Ms, N_test=N_test, testdata=testdata)
                plot_regression(x_test, y_test, mu, sigma, Ms)
            else:
                predictions_matrix, pred_individual_list = get_mimo_predictions(model_path = model_path, Ms = Ms, N_test = N_test, testdata = testdata)
    
    elif mode == 'Classification':
        _, _, testdata = load_cifar("data/")

        if naive == True:
            predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix = get_C_naive_predictions(model_path=model_path, Ms=Ms, testdata = testdata, N_test = 10000, device=device)
            for i in range(len(model_path)):
                np.save(f'reports/Logs/{model_name}/Naive_M{Ms[i]}_predictions', predictions_matrix[i,:])
                np.save(f'reports/Logs/{model_name}/Naive_M{Ms[i]}_confidences', confidences_matrix[i,:])
                np.save(f'reports/Logs/{model_name}/Naive_M{Ms[i]}_correct_predictions', correct_preds_matrix[i,:])

        else:
            predictions_matrix, pred_individual_list, confidences_matrix, correct_preds_matrix = get_C_mimo_predictions(model_path=model_path, Ms=Ms, testdata = testdata, N_test = 10000, device=device)
        
            for i in range(len(model_path)):
                np.save(f'reports/Logs/{model_name}/M{Ms[i]}_predictions', predictions_matrix[i,:])
                np.save(f'reports/Logs/{model_name}/M{Ms[i]}_confidences', confidences_matrix[i,:])
                np.save(f'reports/Logs/{model_name}/M{Ms[i]}_correct_predictions', correct_preds_matrix[i,:])
    print('hej')

    
