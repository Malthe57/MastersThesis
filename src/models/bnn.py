import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

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

        return torch.log(self.pi * prob1 + ((1 - self.pi) * prob2)).sum() if self.sigma2.item() > 0 else torch.log(prob1).sum()
    
class Gaussian():
    def __init__(self, mu, rho, device='cpu'):
        self.device = device
        self.mu = mu
        self.rho = rho
        # self.normal = torch.distributions.Normal(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
        self.init_distribution()

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def init_distribution(self):
        self.normal = torch.distributions.Normal(self.mu, self.sigma)
    
    def rsample(self):
        self.init_distribution()
        return self.normal.rsample()
    
    def log_prob(self, w):
        return self.normal.log_prob(w).sum()

    # def rsample(self):
    #     epsilon = self.normal.sample(self.rho.size())

    #     return self.mu + self.sigma * epsilon
    
    # def log_prob(self, w):
    #     return (-torch.log(torch.sqrt(torch.tensor(2 * np.pi)))
    #             - torch.log(self.sigma)
    #             - ((w - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class BayesianLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.tensor(0.3), device='cpu'):
        super().__init__()
        """
        """        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # initialise mu and rho parameters so they get updated in backpropagation
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        # self.weight_rho = nn.Parameter(torch.Tensor(output_dim, input_dim))
        # self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-6, -5))
        self.weight_rho = nn.Parameter(torch.Tensor(output_dim, input_dim).uniform_(-6, -5)) 
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        # self.bias_rho = nn.Parameter(torch.Tensor(output_dim))
        # self.bias_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-6, -5))
        self.bias_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(-6, -5))

        self.init_mu_weights()
        # self.init_rho_weights()

        # initialise priors
        self.weight_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)
        self.bias_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)

        # initialise variational posteriors
        self.weight_posterior = Gaussian(self.weight_mu, self.weight_rho, device=device)
        self.bias_posterior = Gaussian(self.bias_mu, self.bias_rho, device=device)

        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

        

    def init_mu_weights(self):
        """
        init mu weights like regular nn.Conv2d layers
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L143
        """
        k = self.weight_mu.size(1) # input_features
        nn.init.uniform_(self.weight_mu, -(1/math.sqrt(k)), 1/math.sqrt(k))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_mu, -bound, bound)

    def init_rho_weights(self):
        """
        init rho weights like regular nn.Conv2d layers
        """
        k = self.weight_rho.size(1)
        nn.init.uniform_(self.weight_rho, -(1/math.sqrt(k)), 1/math.sqrt(k))
        if self.bias_rho is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_rho)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_rho, -bound, bound)

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

        output = F.linear(x, w, b)

        return output

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, hidden_units1, hidden_units2, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu", input_dim=1):
        super().__init__()
        """
        """
        self.layer1 = BayesianLinearLayer(input_dim, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, hidden_units2, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer3 = BayesianLinearLayer(hidden_units2, 2, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)

        self.layers = [self.layer1, self.layer2, self.layer3]

        self.device = device

    def forward(self, x, sample=True):
        x = F.relu(self.layer1(x, sample))
        x = F.relu(self.layer2(x, sample))
        x = self.layer3(x, sample)

        mu = x[:, 0]
        sigma = self.get_sigma(x[:, 1])

        return mu, sigma
    
    def inference(self, x, sample=True, n_samples=10):
        # log_probs : (n_samples, batch_size)
        mus = np.zeros((n_samples, x.size(0)))
        sigmas = np.zeros((n_samples, x.size(0)))

        for i in range(n_samples):
            mu, sigma = self.forward(x, sample)
            mus[i] = mu
            sigmas[i] = sigma

        mus = torch.tensor(mus)
        sigmas = torch.tensor(sigmas)
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

    def compute_ELBO(self, input, target, weight, n_samples=1, val = False):

        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            mu, sigma = self.forward(input, sample=True)
            # sigma = self.get_sigma(rho)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(mu, target, sigma)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = (weight*(log_variational_posterior - log_prior)) + NLL

        return loss, log_prior, log_variational_posterior, NLL, mu

class BayesianConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, device='cpu', pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.tensor(0.3)):
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
        # self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size).uniform_(-6, -5))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size).uniform_(-6, -5))
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        # self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        # self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-6, -5))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-6, -5))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        # self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.init_mu_weights()
        # self.init_rho_weights()

        # initialise priors
        self.weight_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)
        self.bias_prior = ScaleMixturePrior(pi, sigma1, sigma2, device=device)

        # initialise variational posteriors
        self.weight_posterior = Gaussian(self.weight_mu, self.weight_rho, device=device)
        self.bias_posterior = Gaussian(self.bias_mu, self.bias_rho, device=device)

    def init_mu_weights(self):
        """
        init mu weights like regular nn.Conv2d layers
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L143
        """
        k = self.weight_mu.size(1) * np.prod(self.kernel_size)
        nn.init.uniform_(self.weight_mu, -(1/math.sqrt(k)), 1/math.sqrt(k))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_mu, -bound, bound)

    def init_rho_weights(self):
        """
        init rho weights like regular nn.Conv2d layers
        """
        k = self.weight_rho.size(1) * np.prod(self.kernel_size)
        nn.init.uniform_(self.weight_rho, -(1/math.sqrt(k)), 1/math.sqrt(k))
        if self.bias_rho is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_rho)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_rho, -bound, bound)

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
    def __init__(self, hidden_units1=128, channels1=64, channels2=128, channels3=256, n_classes=10, pi=0.5, sigma1=torch.exp(torch.tensor(0)), sigma2=torch.exp(torch.tensor(-6)), device="cpu"):
        super().__init__()
        """
        """
        self.conv1 = BayesianConvLayer(3, channels1, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv2 = BayesianConvLayer(channels1, channels2, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv3 = BayesianConvLayer(channels2, channels3, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.conv4 = BayesianConvLayer(channels3, channels3, kernel_size=(3,3), padding=1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer1 = BayesianLinearLayer(channels3*32*32, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer12 = BayesianLinearLayer(hidden_units1, hidden_units1, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.layer2 = BayesianLinearLayer(hidden_units1, n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.layer1, self.layer12, self.layer2]

        self.device = device

    def forward(self, x, sample=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.layer1(x, sample=True))
        x = F.relu(self.layer12(x, sample=True))
        x = self.layer2(x, sample=True)

        log_probs = F.log_softmax(x, dim=1)
        x = torch.argmax(log_probs, dim=1)

        return x, log_probs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, batch_size, n_classes)
        log_probs_matrix = np.zeros((n_samples, x.size(0), n_classes))

        for i in range(n_samples):
            pred, log_probs = self.forward(x, sample)
            log_probs_matrix[i] = log_probs.cpu().detach().numpy()

        mean_log_probs = log_probs_matrix.mean(0)
        mean_predictions = np.argmax(mean_log_probs, axis=1)

        return mean_predictions, mean_log_probs

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
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        NLL = loss_fn(pred, target)
        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, weight, n_samples=1, val = False):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            # pred, probs = self.forward(input, sample=True)
            pred, probs = self.forward(input)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = (weight*(log_variational_posterior - log_prior)) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, pred

class BayesianWideBlock(nn.Module):
    """"
    Basic wide block used in Wide ResNet, but built with Bayesian NNs. It consists of two convolutional layers (with dropout in between) 
    and a skip connection. 

    Inputs:
    - in_channels: number of input channels
    - out_channels: number of output channels
    - stride: stride of the first convolutional layer
    - p: dropout probability

    Returns:
    - out: output tensor
    """
    def __init__(self, in_channels, out_channels, p=0.3, stride=1, device='cpu'):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = BayesianConvLayer(in_channels, out_channels, kernel_size=(3,3), padding=1, device=device)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, device=device)
        self.dropout = nn.Dropout(p=p)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = BayesianConvLayer(out_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, device=device)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, device=device)

        # skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device),
                BayesianConvLayer(in_channels, out_channels, kernel_size=(1,1), stride=stride, device=device)
            )

    def forward (self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.skip(x)

        return out
    
class BayesianWideResnet(nn.Module):
    """
    Bayesian Wide ResNet model for classification. Code adapted from https://github.com/meliketoy/wide-resnet.pytorch/tree/master
    """
    def __init__(self, depth, widen_factor, dropout_rate, n_classes=10, device='cpu'):
        super().__init__()
        self.in_channels = 16
        self.device = device
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.layer1 = self.conv3x3(3, nStages[0])
        self.layer2 = self._wide_layer(BayesianWideBlock, nStages[1], n, dropout_rate, stride=1, device=device)
        self.layer3 = self._wide_layer(BayesianWideBlock, nStages[2], n, dropout_rate, stride=2, device=device)
        self.layer4 = self._wide_layer(BayesianWideBlock, nStages[3], n, dropout_rate, stride=2, device=device)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = BayesianLinearLayer(nStages[3], n_classes, device=device)

    def conv3x3(self, in_channels, out_channels, stride=1):
        # return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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

        # Log-softmax over dimension 1 (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=1)(out)
        x = torch.argmax(log_probs, dim=1)

        return x, log_probs
    
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
    
    def compute_NLL(self, pred, target):
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        NLL = loss_fn(pred, target)
        return NLL
    
    def compute_ELBO(self, input, target, weight, n_samples=1, val=False):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            pred, probs = self.forward(input)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target)

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = (weight*(log_variational_posterior - log_prior))*0 + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, pred

