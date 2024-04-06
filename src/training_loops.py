import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

#define training functions
#train loop for MIMO regression with MSE-loss
def train_regression(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10):

    losses = []
    val_losses = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            model.train()

            x_,y_ = x_.float(), y_.float()

            optimizer.zero_grad()

            output, individual_outputs = model(x_)
            loss = nn.functional.mse_loss(individual_outputs, y_)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  
            wandb.log({"Train loss": loss.item()})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float(), val_y.float()
                    val_output, val_individual_outputs = model(val_x)
                    val_loss = nn.functional.mse_loss(val_individual_outputs, val_y)
                    val_loss_list.append(val_loss.item())
                    wandb.log({"Val loss": val_loss.item()})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")

        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)

    return losses, val_losses

#Train loop for MIMO regression with NLL-loss
def train_var_regression(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10):

    losses = []
    val_losses = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            model.train()

            x_,y_ = x_.float(), y_.float()

            optimizer.zero_grad()

            mu, sigma, mus, sigmas = model(x_)
            loss = torch.nn.GaussianNLLLoss(reduction='mean')(mus, y_, sigmas.pow(2))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  
            wandb.log({"Train loss": loss.item()})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float(), val_y.float()
                    val_mu, val_sigma, val_mus, val_sigmas = model(val_x)
                    val_loss = torch.nn.GaussianNLLLoss(reduction='mean')(val_mus, val_y, val_sigmas.pow(2))
                    val_loss_list.append(val_loss.item())
                    wandb.log({"Val loss": val_loss.item()})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/regression/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)

    return losses, val_losses

#train loop for Bayesian Regression
def train_BNN(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='BNN', val_every_n_epochs=10, device='cpu'):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")


    losses = []
    log_priors = []
    log_variational_posteriors = []
    NLLs = []

    val_losses = []
    val_log_priors = []
    val_log_variational_posteriors = []
    val_NLLs = []

    best_val_loss = np.inf

    num_batches_train = len(trainloader.dataset) // trainloader.batch_size
    num_batches_val = len(valloader.dataset) // valloader.batch_size

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            model.train()

            x_, y_ = x_.float().to(device), y_.float().to(device)

            optimizer.zero_grad()

            loss, log_prior, log_posterior, log_NLL = model.compute_ELBO(x_, y_, num_batches_train)
            
            loss.backward(retain_graph=False)
            optimizer.step()

            losses.append(loss.item()) 
            log_priors.append(log_prior.item())
            log_variational_posteriors.append(log_posterior.item())
            NLLs.append(log_NLL.item()) 
            wandb.log({"Train loss": loss.item(),
                      "Train log_prior": log_prior.item(),
                      "Train log_posterior": log_posterior.item(),
                      "Train log_NLL": log_NLL.item()})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float().to(device), val_y.float().to(device)
                
                    val_loss, _ , _, _, _ = model.compute_ELBO(val_x, val_y, num_batches_val)
                    val_loss_list.append(val_loss.item())
                    wandb.log({"Val loss": val_loss.item()})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/regression/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)
    return losses, log_priors, log_variational_posteriors, NLLs, val_losses

#train loop for Baseline and MIMO classification
def train_classification(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu'):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")
    
    losses = []
    val_losses = []
    val_checkpoint_list = []

    best_val_loss = np.inf
    loss_fn = nn.NLLLoss(reduction='mean')

    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            x_,y_ = x_.float().to(device), y_.long().to(device)

            model.train()

            optimizer.zero_grad()

            log_prob, _, _ = model(x_)
            
            # sum loss per subnetwork
            # mean is already taken over the batch, because we use reduction = 'mean' in the loss function
            loss = 0


            for log_p, y in zip(log_prob, y_.T):
                # print(log_p.shape)
                # print(y.shape)
                loss += loss_fn(log_p, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            wandb.log({"Train loss": loss.item()})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float().to(device), val_y.long().to(device)
                    log_prob, output, _ = model(val_x)
                    val_preds.extend(list(output.cpu().detach().numpy()))
                    val_targets.extend(list(val_y[:,0].cpu().detach().numpy()))

                    val_loss = 0
                    for log_p, y in zip(log_prob, val_y.T):
                        val_loss += loss_fn(log_p, y)

                    val_loss_list.append(val_loss.item())
                    wandb.log({"Val loss": val_loss.item()})

                if (e+1) % checkpoint_every_n_epochs == 0:
                    val_checkpoint_list.append(log_prob)

            val_accuracy = (np.array(val_preds) == np.array(val_targets)).mean()
            wandb.log({"Val accuracy": val_accuracy})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/classification/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)

    torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')

    return losses, val_losses, val_checkpoint_list

#train loop for Bayesian classification
def train_BNN_classification(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='C_BNN', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu'):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")


    losses = []
    log_priors = []
    log_variational_posteriors = []
    NLLs = []

    val_losses = []
    val_log_priors = []
    val_log_variational_posteriors = []
    val_NLLs = []

    val_checkpoint_list = []

    best_val_loss = np.inf

    num_batches_train = len(trainloader.dataset) // trainloader.batch_size
    num_batches_val = len(valloader.dataset) // valloader.batch_size


    for e in tqdm(range(epochs)):
        
        for x_, y_ in trainloader:

            x_, y_ = x_.float().to(device), y_.type(torch.LongTensor).to(device)

            model.train()

            optimizer.zero_grad()

            loss, log_prior, log_posterior, log_NLL, _ = model.compute_ELBO(x_, y_, num_batches_train)
 
            loss.backward()
            optimizer.step()

            losses.append(loss.item()) 
            log_priors.append(log_prior.item())
            log_variational_posteriors.append(log_posterior.item())
            NLLs.append(log_NLL.item()) 
            
            wandb.log({"Train loss": loss.item(),
            "Train log_prior": log_prior.item(),
            "Train log_posterior": log_posterior.item(),
            "Train log_NLL": log_NLL.item()})

        if (e) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float().to(device), val_y.type(torch.LongTensor).to(device)
                
                    val_loss, val_log_prior, val_log_posterior, val_NLL, log_prob = model.compute_ELBO(val_x, val_y, num_batches_val)
                    forward_call = model.forward(val_x, sample=True)
                    val_preds.extend(list(forward_call[0].cpu().detach().numpy()))

                    if len(val_y.shape) > 1:
                        val_targets.extend(list(val_y[:,0].cpu().detach().numpy()))
                    else:
                        val_targets.extend(list(val_y.cpu().detach().numpy()))

                    val_loss_list.append(val_loss.item())
                    wandb.log({"Val loss": val_loss.item(),
                               "Val log_prior": val_log_prior.item(),
                               "Val log_posterior": val_log_posterior.item(),
                               "Val log_NLL": val_NLL.item()})

                if (e) % checkpoint_every_n_epochs == 0:
                    val_checkpoint_list.append(log_prob)

            val_accuracy = (np.array(val_preds) == np.array(val_targets)).mean()
            wandb.log({"Val accuracy": val_accuracy})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/classification/{model_name}.pt')
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)
        
    torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')
    
    return losses, log_priors, log_variational_posteriors, NLLs, val_losses