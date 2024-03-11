import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.bnn import BayesianLinearLayer, ScaleMixturePrior, Gaussian, BayesianConvLayer

class MIMBONeuralNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1, hidden_units2, device="cpu"):
        super().__init__()
        """
        """
        self.n_subnetworks = n_subnetworks
        self.layer1 = BayesianLinearLayer(n_subnetworks, hidden_units1, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, hidden_units2, device=device)
        self.layer3 = BayesianLinearLayer(hidden_units2, 2*n_subnetworks, device=device)

        self.layers = [self.layer1, self.layer2, self.layer3]

        self.device = device
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x, sample=True):
        x = F.relu(self.layer1(x, sample))
        x = F.relu(self.layer2(x, sample))
        x = self.layer3(x, sample)

        mus = x[:, :self.n_subnetworks]
        sigmas = self.get_sigma(x[:, self.n_subnetworks:])

        # get mu and sigma for MIMO mixture
        mu = torch.mean(mus, dim=1)
        sigma = (torch.mean((mus.pow(2) + sigmas.pow(2)), dim=1) - mu.pow(2)).sqrt()

        return mu, sigma, mus, sigmas
    
    def inference(self, x, sample=True, n_samples=10):
        # log_probs : (n_samples, batch_size)
        mus = torch.zeros((n_samples, x.size(0)))
        sigmas = torch.zeros((n_samples, x.size(0)))

        for i in range(n_samples):
            mu, sigma, _, _ = self.forward(x, sample)
            mus[i] = mu
            sigmas[i] = sigma

        # get mu and sigma for sample mixture
        expected_mu = torch.mean(mus, dim=0)
        expected_sigma = (torch.mean((mus.pow(2) + sigmas.pow(2)), dim=0) - expected_mu.pow(2)).sqrt()
    
        return expected_mu, expected_sigma, mus, sigmas


    def compute_log_prior(self):
        model_log_prior = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinearLayer):
                model_log_prior += layer.log_prior

        return model_log_prior
       

    def compute_log_variational_posterior(self):
        model_log_variational_posterior = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinearLayer):
                model_log_variational_posterior += layer.log_variational_posterior
        return model_log_variational_posterior
    
    def compute_NLL(self, mu, target, sigma):
        loss_fn = torch.nn.GaussianNLLLoss(reduction='sum', eps=1e-6)
        var = torch.pow(sigma, 2)
        NLL = loss_fn(mu, target, var)
        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, num_batches, n_samples=1):

        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            mu, sigma, mus, sigmas = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(mus, target, sigmas) 

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL

        return loss, log_prior, log_variational_posterior, NLL
    
class MIMBOConvNeuralNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1=128, channels1=32, channels2=64, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu"):
        super().__init__()
        """
        """
        self.n_subnetworks = n_subnetworks
        self.conv1 = BayesianConvLayer(3*n_subnetworks, channels1, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv2 = BayesianConvLayer(channels1, channels2, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv3 = BayesianConvLayer(channels2, channels2, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer1 = BayesianLinearLayer(channels2*32*32, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, n_subnetworks*10, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)

        
        self.layers = [self.conv1, self.conv2, self.conv3, self.layer1, self.layer2]

        self.device = device

    def forward(self, x, sample=True):
        # put the input through the conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.relu(self.conv3(x, sample))
        # reshape to fit into linear layer
        x = x.reshape(x.size(0),-1)
        # put the input through the linear layers
        x = F.relu(self.layer1(x, sample))
        x = self.layer2(x, sample)
        
        # reshape to batch_size x M x 10
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax (because we are using NLLloss) over the class dimension 
        x = nn.LogSoftmax(dim=2)(x)
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(x, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(x, dim=1).argmax(dim=1) # dim : batch_size
        
        x = x.permute(1,0,2)

        return x, output, individual_outputs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, batch_size, n_classes)
        log_probs = np.zeros((n_samples, self.n_subnetworks, x.size(0),  n_classes))

        for i in range(n_samples):
            probs, output, individual_outputs = self.forward(x, sample)
            log_probs[i] = probs.cpu().detach().numpy()

        mean_subnetwork_probs = np.mean(log_probs, axis=1)
        mean_probs = np.mean(mean_subnetwork_probs, axis=0)

        mean_predictions = np.argmax(mean_probs, axis=1)

        return mean_predictions, mean_subnetwork_probs, mean_probs

    def compute_log_prior(self):
        model_log_prior = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_prior += layer.log_prior
        return model_log_prior

    def compute_log_variational_posterior(self):
        model_log_variational_posterior = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_variational_posterior += layer.log_variational_posterior
        return model_log_variational_posterior
    
    def compute_NLL(self, pred, target):
        NLL = 0
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        for p, t in zip(pred, target.T):
            NLL += F.nll_loss(p, t)

        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, num_batches, n_samples=1):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            probs, output, individual_outputs= self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL

