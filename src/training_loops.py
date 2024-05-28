import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from utils.utils import logmeanexp

def destandardise(min, max, y):
    '''
    Destandardise outputs from standardised model.
    Inputs:
    - min: the min value used for standardisation of data
    - max: the max value used for standardisation of data
    - y: the data to be destandardised
    
    Outputs:
    - y: destandardised data
    '''
    if min is None or max is None:
        return y

    y = ((y + 1)/2*(max - min) + min)
    return y

def minibatch_weighting(dataloader, target):
    return target.shape[0] / len(dataloader.dataset)

def blundell_minibatch_weighting(dataloader, i):
    num_batches = len(dataloader)
    
    weight = 2**(num_batches - i) / ((2**num_batches) - 1) # from Blundell et al. 2015

    return weight

def get_init_checkpoint(model, valloader, device):
    checkpoint = None
    model.eval()
    with torch.no_grad():
        for k, (val_x, val_y) in enumerate(valloader, 1):
            val_x, val_y = val_x.float().to(device), val_y.float().to(device)
            log_prob, _, _ = model(val_x)
            if k == 1:
                checkpoint = log_prob

    return checkpoint

def get_init_checkpoint_BNN(model, valloader, device):
    checkpoint = None
    model.eval()
    with torch.no_grad():
        for k, (val_x, val_y) in enumerate(valloader, 1):
            val_x, val_y = val_x.float().to(device), val_y.type(torch.LongTensor).to(device)
            val_weight = blundell_minibatch_weighting(valloader, k)
            if len(val_y.shape) > 1:
                val_y = val_y[:,0]
            _, _, _, _, log_prob, _ = model.compute_ELBO(val_x, val_y, val_weight, val=True)
            if k == 1:
                checkpoint = log_prob

    return checkpoint
        
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
def train_var_regression(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10, device='cpu', save=True, **kwargs):
    
    # if kwargs is not None:
    #     max = kwargs['max']
    #     min = kwargs['min']

    losses = []
    val_losses = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        train_preds = []
        train_targets = []

        for x_, y_ in trainloader:

            model.train()

            x_,y_ = x_.float().to(device), y_.float().to(device)

            optimizer.zero_grad()

            mu, sigma, mus, sigmas = model(x_)
            loss = torch.nn.GaussianNLLLoss(reduction='sum')(mus, y_, sigmas.pow(2))
            train_preds.extend(list(mu.cpu().detach().numpy()))
            train_targets.extend(list(y_[:,0].cpu().detach().numpy()))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())  
            wandb.log({"Train loss": loss.item()})

        train_RMSE = np.atleast_1d(np.sqrt(np.mean((np.array(train_preds) - np.array(train_targets))**2)))
        for i in range(len(train_RMSE)):
            wandb.log({f"Train RMSE {i}": train_RMSE[i]})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for val_x, val_y in valloader:
                    val_x, val_y = val_x.float(), val_y.float()
                    val_mu, val_sigma, val_mus, val_sigmas = model(val_x)
                    if len(val_y.shape) > 1:
                        val_loss = torch.nn.GaussianNLLLoss(reduction='sum')(val_mu, val_y[:,0], val_sigma.pow(2))
                    else:
                        val_loss = torch.nn.GaussianNLLLoss(reduction='sum')(val_mu, val_y, val_sigma.pow(2))
                    val_preds = val_mu.cpu().detach().numpy()
                    val_targets = val_y[:,0].cpu().detach().numpy()
                    val_loss_list.append(val_loss.item())
                    
                    # wandb.log({"Val loss": val_loss.item()})
            
            val_RMSE = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2))
            wandb.log({"Val RMSE": val_RMSE})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            wandb.log({"Val loss": mean_val_loss})

            

            if mean_val_loss < best_val_loss and save:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/regression/{model_name}.pt')
        
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)

    return losses, val_losses

#train loop for Bayesian Regression
def train_BNN(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='BNN', val_every_n_epochs=10, device='cpu', save=True, **kwargs):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")

    # if kwargs is not None:
    #     max = kwargs['max']
    #     min = kwargs['min']


    losses = []
    log_priors = []
    log_variational_posteriors = []
    NLLs = []

    val_losses = []
    val_log_priors = []
    val_log_variational_posteriors = []
    val_NLLs = []

    best_val_loss = np.inf

    for e in tqdm(range(epochs)):
        
        train_preds = []
        train_targets = []

        for i, (x_, y_) in enumerate(trainloader, 1):

            model.train()

            x_, y_ = x_.float().to(device), y_.float().to(device)

            optimizer.zero_grad()

            train_weight = blundell_minibatch_weighting(trainloader, i)
            loss, log_prior, log_posterior, log_NLL, pred = model.compute_ELBO(x_, y_, train_weight)
            
            loss.backward(retain_graph=False)
            optimizer.step()

            train_preds.extend(list(pred.cpu().detach().numpy()))
            train_targets.extend(list(y_.cpu().detach().numpy()))

            losses.append(loss.item()) 
            log_priors.append(log_prior.item())
            log_variational_posteriors.append(log_posterior.item())
            NLLs.append(log_NLL.item()) 
            wandb.log({"Train loss": loss.item(),
                      "Train log_prior": log_prior.item(),
                      "Train log_posterior": log_posterior.item(),
                      "Train log_NLL": log_NLL.item()})
            
        train_RMSE = np.atleast_1d(np.sqrt(np.mean((np.array(train_preds) - np.array(train_targets))**2)))

        for j in range(len(train_RMSE)):
            wandb.log({f"Train RMSE {j}": train_RMSE[j]})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for k, (val_x, val_y) in enumerate(valloader, 1):
                    val_x, val_y = val_x.float().to(device), val_y.float().to(device)

                    val_weight = blundell_minibatch_weighting(valloader, k)
                    if len(val_y.shape) > 1: # MIMBO
                        
                        val_loss, _ , _, _, pred = model.compute_ELBO(val_x, val_y[:,0], val_weight, val=True)
                        val_preds.extend(list(pred.cpu().detach().numpy()))
                        val_targets.extend(list(val_y[:,0].cpu().detach().numpy()))

                    else: # BNN
                        val_loss, _ , _, _, pred = model.compute_ELBO(val_x, val_y, val_weight, val=True)
                        val_preds.extend(list(pred.cpu().detach().numpy()))
                        val_targets.extend(list(val_y.cpu().detach().numpy()))
                
                    val_loss_list.append(val_loss.item())
                    # wandb.log({"Val loss": val_loss.item()})

            val_RMSE = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2))
            wandb.log({"Val RMSE": val_RMSE})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            wandb.log({"Val loss": mean_val_loss})

            if mean_val_loss < best_val_loss and save:
                best_val_loss = mean_val_loss
                torch.save(model, f'models/regression/{model_name}.pt')
            
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        # after every epoch, step the scheduler
        scheduler.step(mean_val_loss)

    return losses, log_priors, log_variational_posteriors, NLLs, val_losses

#train loop for Baseline and MIMO classification
def train_classification(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu', save=True):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")

    patience = 0
    
    losses = []
    val_losses = []
    val_checkpoint_list = [get_init_checkpoint(model, valloader, device)]

    best_val_loss = np.inf
    best_val_acc = 0
    loss_fn = nn.NLLLoss(reduction='sum')



    for e in tqdm(range(epochs)):

        train_preds = []
        train_targets = []
        
        for x_, y_ in trainloader:

            x_,y_ = x_.float().to(device), y_.long().to(device)

            model.train()

            optimizer.zero_grad()

            log_prob, _, individual_pred = model(x_)

            train_preds.extend(list(individual_pred.cpu().detach().numpy()))
            train_targets.extend(list(y_.cpu().detach().numpy()))
            
            loss = loss_fn(log_prob, y_)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            wandb.log({"Train loss": loss.item()})

        
        train_accuracy = np.atleast_1d((np.array(train_preds) == np.array(train_targets)).mean(0))
        
        for i in range(len(train_accuracy)):
            wandb.log({f"Train accuracy {i}": train_accuracy[i]})

        if (e+1) % val_every_n_epochs == 0:
            model.eval()

            val_loss_list = []
            val_preds = []
            val_targets = []
            val_checkpoint = []
            with torch.no_grad():
                for k, (val_x, val_y) in enumerate(valloader,1):
                    val_x, val_y = val_x.float().to(device), val_y.long().to(device)
                    log_prob, output, _ = model(val_x)
                    val_preds.extend(list(output.cpu().detach().numpy()))
                    val_targets.extend(list(val_y[:,0].cpu().detach().numpy()))

                    # mean over n_subnetworks 
                    log_p = torch.log(torch.exp(log_prob).mean(2))
                    val_loss = loss_fn(log_p, val_y[:,0])

                    val_loss_list.append(val_loss.item())
                    # wandb.log({"Val loss": val_loss.item()})
                    if k==1:
                        val_checkpoint = log_prob

                if (e+1) % checkpoint_every_n_epochs == 0:
                    val_checkpoint_list.append(val_checkpoint)

            val_accuracy = (np.array(val_preds) == np.array(val_targets)).mean()
            wandb.log({"Val accuracy": val_accuracy})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            wandb.log({"Val loss": mean_val_loss})

            if val_accuracy > best_val_acc and save:
                best_val_acc = val_accuracy
                torch.save(model, f'models/classification/{model_name}.pt')
                patience = 0
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        # scheduler.step(mean_val_loss)
        # scheduler.step(val_accuracy)
        scheduler.step()

        patience += 1
        
        # if patience > 10:
        #     break

    if save:
        torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')

    return losses, val_losses, val_checkpoint_list

#train loop for Bayesian classification
def train_BNN_classification(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='C_BNN', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu', save=True):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")

    patience = 0

    losses = []
    log_priors = []
    log_variational_posteriors = []
    NLLs = []

    val_losses = []

    val_checkpoint_list = [get_init_checkpoint_BNN(model, valloader, device)]

    best_val_acc = 0

    for e in tqdm(range(epochs)):

        train_preds = []
        train_targets = []
        
        for i, (x_, y_) in enumerate(trainloader, 1): # start enumeration at 1

            x_, y_ = x_.float().to(device), y_.type(torch.LongTensor).to(device)

            model.train()

            optimizer.zero_grad()

            train_weight = blundell_minibatch_weighting(trainloader, i)
            loss, log_prior, log_posterior, log_NLL, _, pred = model.compute_ELBO(x_, y_, train_weight)

            loss.backward()
            optimizer.step()

            train_preds.extend(list(pred.cpu().detach().numpy()))
            train_targets.extend(list(y_.cpu().detach().numpy()))


            losses.append(loss.item()) 
            log_priors.append(log_prior.item())
            log_variational_posteriors.append(log_posterior.item())
            NLLs.append(log_NLL.item()) 
            
            wandb.log({"Train loss": loss.item(),
            "Train log_prior": log_prior.item(),
            "Train log_posterior": log_posterior.item(),
            "Train log_NLL": log_NLL.item()})

 
        train_accuracy = np.atleast_1d((np.array(train_preds) == np.array(train_targets)).mean(0))
        
        for j in range(len(train_accuracy)):
            wandb.log({f"Train accuracy {j}": train_accuracy[j]})

        if (e) % val_every_n_epochs == 0:
            model.eval()

            val_checkpoint = []
            val_loss_list = []
            val_log_prior_list = []
            val_log_posterior_list = []
            val_NLL_list = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for k, (val_x, val_y) in enumerate(valloader, 1):
                    val_x, val_y = val_x.float().to(device), val_y.type(torch.LongTensor).to(device)
                
                    val_weight = blundell_minibatch_weighting(valloader, k)
                    val_loss, val_log_prior, val_log_posterior, val_NLL, log_prob, pred = model.compute_ELBO(val_x, val_y, val_weight, val=True)

                    if len(val_y.shape) > 1:
                        val_preds.extend(list(pred.cpu().detach().numpy()))
                        val_targets.extend(list(val_y[:,0].cpu().detach().numpy()))
                    else:
                        val_preds.extend(list(pred.cpu().detach().numpy()))
                        val_targets.extend(list(val_y.cpu().detach().numpy()))

                    val_loss_list.append(val_loss.item())
                    val_log_prior_list.append(val_log_prior.item())
                    val_log_posterior_list.append(val_log_posterior.item())
                    val_NLL_list.append(val_NLL.item())
                    # wandb.log({"Val loss": val_loss.item(),
                    #            "Val log_prior": val_log_prior.item(),
                    #            "Val log_posterior": val_log_posterior.item(),
                    #            "Val log_NLL": val_NLL.item()})
                    if k == 1:
                        val_checkpoint = log_prob

                if (e) % checkpoint_every_n_epochs == 0:
                    val_checkpoint_list.append(val_checkpoint)

            val_accuracy = (np.array(val_preds) == np.array(val_targets)).mean(0)
            wandb.log({"Val accuracy": val_accuracy})

            val_losses.extend(val_loss_list)
            mean_val_loss = np.mean(val_loss_list)
            mean_log_prior = np.mean(val_log_prior_list)
            mean_log_posterior = np.mean(val_log_posterior_list)
            mean_NLL = np.mean(val_NLL_list)
            wandb.log({"Val loss": mean_val_loss,
                            "Val log_prior": mean_log_prior,
                            "Val log_posterior": mean_log_posterior,
                            "Val log_NLL": mean_NLL})

            if val_accuracy > best_val_acc and save:
                best_val_acc = val_accuracy
                torch.save(model, f'models/classification/{model_name}.pt')
                patience = 0
            # print(f"Mean validation loss at epoch {e}: {mean_val_loss}")
                
        # after every epoch, step the scheduler
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        # scheduler.step(mean_val_loss)
        scheduler.step(val_accuracy)
        # scheduler.step()

        patience += 1
        # if patience > 10:
        #     break
    
    if save:
        torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')
    
    return losses, log_priors, log_variational_posteriors, NLLs, val_losses

def train(trainloader, model, criterion, optimizer, scheduler, device):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(trainloader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        log_probs, output, individual_outputs = model(input)
        loss = criterion(log_probs, target)

        # measure accuracy and record loss
        prec1 = accuracy(individual_outputs.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"Train loss": loss.item()})
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        acc = np.atleast_1d(prec1.cpu().detach().numpy())
        for j in range(len(acc)):
            wandb.log({f"Train accuracy {j}": acc[j]})

def BNN_train(trainloader, model, optimizer, scheduler, device):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(trainloader, 1):
        target = target.type(torch.LongTensor).to(device)
        input = input.to(device)

        # compute output
        train_weight = blundell_minibatch_weighting(trainloader, i)
        loss, log_prior, log_posterior, log_NLL, _, pred = model.compute_ELBO(input, target, train_weight)


        # measure accuracy and record loss
        prec1 = accuracy(pred.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"Train loss": loss.item()})
        wandb.log({"lr": scheduler._last_lr[0]})
        wandb.log({"Train log_prior": log_prior})
        wandb.log({"Train log_posterior": log_posterior})
        wandb.log({"Train log_NLL": log_NLL})
        acc = np.atleast_1d(prec1.cpu().detach().numpy())
        for j in range(len(acc)):
            wandb.log({f"Train accuracy {j}": acc[j]})
        


def validate(valloader, model, criterion, device):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for k, (input, target) in enumerate(valloader,1):
        target = target.type(torch.LongTensor).to(device)[:,0] 
        input = input.to(device)

        # compute output
        with torch.no_grad():
            log_probs, output, individual_outputs = model(input)
        log_p = logmeanexp(log_probs, dim=2)
        loss = criterion(log_p, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if k == 1:
            val_checkpoint = log_probs

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    wandb.log({"Val loss": losses.avg})
    wandb.log({"Val accuracy": top1.avg})

    return top1.avg, val_checkpoint


def BNN_validate(valloader, model, device):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for k, (input, target) in enumerate(valloader,1):
        target = target.type(torch.LongTensor).to(device) if len(target.shape) == 1 else target.type(torch.LongTensor).to(device)[:,0]
        input = input.to(device)

        # compute output
        val_weight = blundell_minibatch_weighting(valloader, k)
        with torch.no_grad():
            val_loss, val_log_prior, val_log_posterior, val_NLL, log_probs, pred = model.compute_ELBO(input, target, val_weight, val=True)


        # measure accuracy and record loss
        prec1 = accuracy(pred.data, target, topk=(1,))[0]
        losses.update(val_loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if k == 1:
            val_checkpoint = log_probs

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    wandb.log({"Val loss": losses.avg})
    wandb.log({"Val accuracy": top1.avg})

    return top1.avg, val_checkpoint


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct.sum(0)
        res.append(correct_k.float().mul_(100.0 / batch_size))
    return res

def run_MIMO_resnet(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='MIMO', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu', save=True):
    
    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")

    val_checkpoint_list = [get_init_checkpoint(model, valloader, device)]

    best_val_acc = 0
    criterion = nn.NLLLoss(reduction='mean')

    for e in tqdm(range(epochs)):
        train(trainloader, model, criterion, optimizer, scheduler, device)

        # evaluate on validation set
        prec1, val_checkpoint = validate(valloader, model, criterion, device)
        if (e+1) % checkpoint_every_n_epochs == 0:
            val_checkpoint_list.append(val_checkpoint)

        scheduler.step()

        if prec1 > best_val_acc and save:
            best_val_acc = prec1
            torch.save(model, f'models/classification/{model_name}.pt')

    if save:
        torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')

def run_BNN_resnet(model, optimizer, scheduler, trainloader, valloader, epochs=500, model_name='C_BNN', val_every_n_epochs=10, checkpoint_every_n_epochs=20, device='cpu', save=True):

    if device == 'cpu':
        print("Training on CPU")
    else:
        print("Cuda available, training on GPU")

    val_checkpoint_list = [get_init_checkpoint_BNN(model, valloader, device)]

    best_val_acc = 0

    for e in tqdm(range(epochs)):
        BNN_train(trainloader, model, optimizer, scheduler, device)

        # evaluate on validation set
        prec1, val_checkpoint = BNN_validate(valloader, model, device)
        if (e+1) % checkpoint_every_n_epochs == 0:
            val_checkpoint_list.append(val_checkpoint)

        scheduler.step()

        if prec1 > best_val_acc and save:
            best_val_acc = prec1
            torch.save(model, f'models/classification/{model_name}.pt')

    if save:
        torch.save(torch.stack(val_checkpoint_list), f'models/classification/checkpoints/{model_name}_checkpoints.pt')
