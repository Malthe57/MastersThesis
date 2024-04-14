import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.bnn import BayesianLinearLayer, ScaleMixturePrior, Gaussian, BayesianConvLayer, BayesianWideBlock
import time

class MIMBONeuralNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1, hidden_units2, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu", input_dim=1):
        super().__init__()
        """
        """
        self.n_subnetworks = n_subnetworks
        self.layer1 = BayesianLinearLayer(input_dim*n_subnetworks, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer3 = BayesianLinearLayer(hidden_units2, 2*n_subnetworks, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)

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

    def compute_ELBO(self, input, target, num_batches, n_samples=1, val = False):

        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            mu, sigma, mus, sigmas = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            if val:
                NLLs[i] = self.compute_NLL(mu, target, sigma)
                output = mu
            else:
                NLLs[i] = self.compute_NLL(mus, target, sigmas) 
                output = mus

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL

        return loss, log_prior, log_variational_posterior, NLL, output
    
class MIMBOConvNeuralNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1=128, channels1=64, channels2=128, channels3=256, n_classes=10, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu"):
        super().__init__()
        """
        """
        self.n_subnetworks = n_subnetworks
        self.conv1 = BayesianConvLayer(3*n_subnetworks, channels1, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv2 = BayesianConvLayer(channels1, channels2, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv3 = BayesianConvLayer(channels2, channels3, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv4 = BayesianConvLayer(channels3, channels3, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer1 = BayesianLinearLayer(channels3*32*32, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, n_subnetworks*n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)

        
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.layer1, self.layer2]

        self.device = device

    def forward(self, x, sample=True):
        # put the input through the conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))

        # reshape to fit into linear layer
        x = x.reshape(x.size(0),-1)
        # put the input through the linear layers
        x = F.relu(self.layer1(x, sample))
        x = self.layer2(x, sample)

        # reshape to batch_size x M x n_classes
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax (because we are using NLLloss) over the class dimension 
        log_probs = nn.LogSoftmax(dim=2)(x) # dim : batch_size x M x n_classes
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size
        
        # log_probs = log_probs.permute(1,0,2) # dim : M x batch_size x n_classes

        return output, individual_outputs, log_probs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, n_subnetworks, batch_size, n_classes)
        # log_probs : (n_samples, batch_size, n_subnetworks, n_classes)
        # log_probs = np.zeros((n_samples, self.n_subnetworks, x.size(0),  n_classes))
        log_probs = np.zeros((n_samples, x.size(0), self.n_subnetworks, n_classes))

        for i in range(n_samples):
            output, individual_outputs, probs = self.forward(x, sample)
            log_probs[i] = probs.cpu().detach().numpy()

        # mean_subnetwork_probs = np.mean(log_probs, axis=1) # mean over n_subnetworks, dim : n_samples x batch_size x n_classes
        # mean_probs = np.mean(mean_subnetwork_probs, axis=0) # mean over samples, dim : batch_size x n_classes
        mean_subnetwork_probs = np.mean(log_probs, axis=2) # mean over n_subnetworks, dim : n_samples x batch_size x n_classes
        mean_probs = np.mean(mean_subnetwork_probs, axis=0) # mean over samples, dim : batch_size x n_classes
        mean_predictions = np.argmax(mean_probs, axis=1) # argmax over n_classes, dim : batch_size

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
    
    def compute_NLL(self, pred, target, val=False):
        if val:
            # mean over n_subnetworks dimension
            NLL = F.nll_loss(pred.mean(0), target[:,0])

        else:
            NLL = 0
            loss_fn = torch.nn.NLLLoss(reduction='sum')
            for p, t in zip(pred, target.T):
                NLL += loss_fn(p, t)

        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, num_batches, n_samples=1, val = False):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            output, individual_outputs, probs = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target, val=val)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, individual_outputs

class MIMBOWideResnet(nn.Module):
    """
    MIMBO Wide ResNet model for classification. Code adapted from https://github.com/meliketoy/wide-resnet.pytorch/tree/master
    """
    def __init__(self, n_subnetworks, depth, widen_factor, dropout_rate, n_classes=10, device='cpu'):
        super().__init__()
        self.in_channels = 16
        self.n_subnetworks = n_subnetworks
        self.device = device
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.layer1 = self.conv3x3(3*n_subnetworks, nStages[0])
        self.layer2 = self._wide_layer(BayesianWideBlock, nStages[1], n, dropout_rate, stride=1, device=device)
        self.layer3 = self._wide_layer(BayesianWideBlock, nStages[2], n, dropout_rate, stride=2, device=device)
        self.layer4 = self._wide_layer(BayesianWideBlock, nStages[3], n, dropout_rate, stride=2, device=device)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = BayesianLinearLayer(nStages[3], n_classes*n_subnetworks, device=device)

    def conv3x3(self, in_channels, out_channels, stride=1):
        return BayesianConvLayer(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, device=self.device)

    def _wide_layer(self, block, out_channels, num_blocks, p, stride, device='cpu'):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, p, stride, device=device))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x, sample=True):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        # reshape to batch_size x M x n_classes
        x = out.reshape(out.size(0), self.n_subnetworks, -1)
        # Log-softmax (because we are using NLLloss) over the class dimension 
        log_probs = nn.LogSoftmax(dim=2)(x) # dim : batch_size x M x n_classes
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size
        
        log_probs = log_probs.permute(1,0,2) # dim : M x batch_size x n_classes

        return output, individual_outputs, log_probs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, n_subnetworks, batch_size, n_classes)
        log_probs = np.zeros((n_samples, self.n_subnetworks, x.size(0),  n_classes))

        for i in range(n_samples):
            output, individual_outputs, probs = self.forward(x, sample)
            log_probs[i] = probs.cpu().detach().numpy()

        mean_subnetwork_probs = np.mean(log_probs, axis=1) # mean over n_subnetworks, dim : n_samples x batch_size x n_classes
        mean_probs = np.mean(mean_subnetwork_probs, axis=0) # mean over samples, dim : batch_size x n_classes

        mean_predictions = np.argmax(mean_probs, axis=1) # argmax over n_classes, dim : batch_size

        return mean_predictions, mean_subnetwork_probs, mean_probs
    
    def compute_log_prior(self):
        model_log_prior = 0.0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.linear]:
            # layer can either be BayesianConvLayer or nn.Sequential() containing 4 Bayesian Blocks
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_prior += layer.log_prior
            elif isinstance(layer, nn.Sequential):
                for block in layer:
                    for module in block.modules():
                        if isinstance(module, BayesianLinearLayer) or isinstance(module, BayesianConvLayer):
                            model_log_prior += module.log_prior
        return model_log_prior
    
    def compute_log_variational_posterior(self):
        model_log_variational_posterior = 0.0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.linear]:
            # layer can either be BayesianConvLayer or nn.Sequential() containing 4 Bayesian Blocks
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_variational_posterior += layer.log_variational_posterior
            elif isinstance(layer, nn.Sequential):
                for block in layer:
                    for module in block.modules():
                        if isinstance(module, BayesianLinearLayer) or isinstance(module, BayesianConvLayer):
                            model_log_variational_posterior += module.log_variational_posterior 
        return model_log_variational_posterior
    
    def compute_NLL(self, pred, target, val=False):
        if val:
            # mean over n_subnetworks dimension
            NLL = F.nll_loss(pred.mean(0), target[:,0])

        else:
            NLL = 0
            loss_fn = torch.nn.NLLLoss(reduction='sum')
            for p, t in zip(pred, target.T):
                NLL += loss_fn(p, t)

        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, num_batches, n_samples=1, val = False):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            output, individual_outputs, probs = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target, val=val)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = ((log_variational_posterior - log_prior) / num_batches) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, individual_outputs