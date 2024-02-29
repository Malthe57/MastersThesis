import torch

def C_BNN_inference(model, testloader, device):

    preds = []
    log_probs = []
    targets = []

    for x_test, y_test in testloader:
        x_test, y_test = x_test.float().to(device), y_test.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred, probs = model.inference(x_test, inference=False, n_samples=10)
            preds.extend(pred)
            log_probs.extend(probs)
            targets.extend(y_test.cpu().detach().numpy())

    return preds, log_probs, targets

if __name__ == "__main__":
    model_names = ['BNN', 'C_BNN']
    model_name = model_names[0]

    modes = ['Regression', 'Classification']
    mode = modes[1]

        


