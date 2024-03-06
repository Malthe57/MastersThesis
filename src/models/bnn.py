import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaleMixturePrior():
    def __init__(self, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.tensor(0.3), device='cpu'):
        self.device = device
        self.pi = pi
        self.mu = 0
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def prob(self, w, sigma):
    
        return (1 / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * torch.pow((w - self.mu), 2) / torch.pow(sigma, 2))

    def log_prob(self, w):
        prob1 = self.prob(w, self.sigma1)
        prob2 = self.prob(w, self.sigma2)

        return torch.log(self.pi * prob1 + (1 - self.pi) * prob2).sum()
    
class Gaussian():
    def __init__(self, mu, rho, device='cpu'):
        self.device = device
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
        
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def rsample(self):
        epsilon = self.normal.sample(self.rho.size()) 
        return self.mu + self.sigma * epsilon

    def log_prob(self, w):
        return self.normal.log_prob(w).sum()

class BayesianLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.tensor(0.3), device='cpu'):
        super().__init__()
        """
        """        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # initialise mu and rho parameters so they get updated in backpropagation
        self.weight_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-5, -4)) 
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(-5, -4))

        # initialise priors
        self.weight_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)
        self.bias_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)

        # initialise variational posteriors
        self.weight_posterior = Gaussian(self.weight_mu, self.weight_rho, device=device)
        self.bias_posterior = Gaussian(self.bias_mu, self.bias_rho, device=device)

        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

    def forward(self, x, sample=True):
        if sample:
            w = self.weight_posterior.rsample()
            b = self.bias_posterior.rsample()

            self.log_prior = self.weight_prior.log_prob(w) + self.bias_prior.log_prob(b)
            self.log_variational_posterior = self.weight_posterior.log_prob(w) + self.bias_posterior.log_prob(b)
            
        else:
            w = self.weight_posterior.mu
            b = self.bias_posterior.mu

            self.log_prior = 0.0
            self.log_variational_posterior = 0.0

        output = torch.mm(x, w) + b

        return output

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, hidden_units1, hidden_units2, device="cpu"):
        super().__init__()
        """
        """
        self.layer1 = BayesianLinearLayer(1, hidden_units1, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, hidden_units2, device=device)
        self.layer3 = BayesianLinearLayer(hidden_units2, 2, device=device)

        self.layers = [self.layer1, self.layer2, self.layer3]

        self.device = device

    def forward(self, x, sample=True):
        x = F.relu(self.layer1(x, sample))
        x = F.relu(self.layer2(x, sample))
        x = self.layer3(x, sample)

        mu = x[:, 0]
        rho = x[:, 1]

        return mu, rho
    
    def inference(self, x, sample=True, n_samples=1):
        # log_probs : (n_samples, batch_size)
        mus = np.zeros((n_samples, x.size(0)))
        sigmas = np.zeros((n_samples, x.size(0)))

        for i in range(n_samples):
            mu, rho = self.forward(x, sample)
            mus[i] = mu
            sigmas[i] = self.get_sigma(rho)

        # parameters for Gaussian mixture
        expected_mu = torch.mean(mus, dim=0)
        expected_sigma = (torch.mean((mus.pow(2) + sigmas.pow(2)), dim=0) - expected_mu.pow(2)).sqrt()
    
        return expected_mu, expected_sigma


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
            mu, rho = self.forward(input, sample=True)
            sigma = self.get_sigma(rho)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(mu, target, sigma)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL

        return loss, log_prior, log_variational_posterior, NLL

class BayesianConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, device='cpu', pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6))):
        super().__init__()
        """
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        
        # initialise mu and rho parameters so they get updated in backpropagation
        # use *kernel_size instead of writing (_, _, kernel_size, kernel_size)
        self.weight_mu = nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))

        # initialise priors
        self.weight_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)
        self.bias_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)

        # initialise variational posteriors
        self.weight_posterior = Gaussian(self.weight_mu.permute(1,0,2,3).to(device), self.weight_rho.permute(1,0,2,3).to(device), device=device)
        self.bias_posterior = Gaussian(self.bias_mu, self.bias_rho, device=device)

    def forward(self, x, sample=True):

        if sample:
            w = self.weight_posterior.rsample()
            b = self.bias_posterior.rsample()

            self.log_prior = self.weight_prior.log_prob(w) + self.bias_prior.log_prob(b)
            self.log_variational_posterior = self.weight_posterior.log_prob(w) + self.bias_posterior.log_prob(b)

        else:
            w = self.weight_posterior.mu
            b = self.bias_posterior.mu

            self.log_prior = 0.0
            self.log_variational_posterior = 0.0

        output = F.conv2d(x, w, b, self.stride, self.padding, self.dilation)

        return output
        
class BayesianConvNeuralNetwork(nn.Module):
    def __init__(self, hidden_units1, channels1, channels2, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu"):
        super().__init__()
        """
        """
        self.conv1 = BayesianConvLayer(3, channels1, kernel_size=(3,3), padding=1, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv2 = BayesianConvLayer(channels1, channels2, kernel_size=(3,3), padding=1, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv3 = BayesianConvLayer(channels2, channels2, kernel_size=(3,3), padding=1, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer1 = BayesianLinearLayer(channels2*32*32, hidden_units1, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, 10, sigma1=sigma1, sigma2=sigma2, device=device)

        
        self.layers = [self.conv1, self.conv2, self.layer1, self.layer2]

        self.device = device

    def forward(self, x, sample=True):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.relu(self.conv3(x, sample))
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.layer1(x, sample))
        x = self.layer2(x, sample)
        probs = F.log_softmax(x, dim=1)
        x = torch.argmax(probs, dim=1)

        return x, probs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, batch_size, n_classes)
        log_probs = np.zeros((n_samples, x.size(0), n_classes))

        for i in range(n_samples):
            pred, probs = self.forward(x, sample)
            log_probs[i] = probs.cpu().detach().numpy()

        mean_log_probs = log_probs.mean(0)
        mean_predictions = np.argmax(mean_log_probs, axis=1)

        return mean_predictions, mean_log_probs

    def compute_log_prior(self):
        model_log_prior = 0.0
        for layer in self.layers:
            if isinstance(layer, (BayesianLinearLayer, BayesianConvLayer)):
                model_log_prior += layer.log_prior
        return model_log_prior

    def compute_log_variational_posterior(self):
        model_log_variational_posterior = 0.0
        for layer in self.layers:
            if isinstance(layer, (BayesianLinearLayer, BayesianConvLayer)):
                model_log_variational_posterior += layer.log_variational_posterior
        return model_log_variational_posterior
    
    def compute_NLL(self, pred, target):
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        NLL = loss_fn(pred, target)
        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, num_batches, n_samples=1):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            pred, probs = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL

