import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd() + '/src/')
from utils.utils import logmeanexp
from models.bnn import ScaleMixturePrior, Gaussian, BayesianLinearLayer, BayesianConvLayer

class BayesianBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, pi=1.0, sigma1=torch.tensor(1.0), sigma2=torch.tensor(0.0), device='cpu'):
        super(BayesianBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = BayesianConvLayer(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                               padding=1, bias=False, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = BayesianConvLayer(out_planes, out_planes, kernel_size=(3,3), stride=1,
                               padding=1, bias=False, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and BayesianConvLayer(in_planes, out_planes, kernel_size=(1,1), stride=stride,
                               padding=0, bias=False, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class BayesianNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(BayesianNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class BayesianWideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.0, n_classes=10, pi=1.0, sigma1=torch.tensor(1.0), sigma2=torch.tensor(0.0), device='cpu'):
        super(BayesianWideResNet, self).__init__()
        print(f"Initializing Bayesian WideResNet")
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BayesianBasicBlock
        # 1st conv before any network block
        self.conv1 = BayesianConvLayer(3, nChannels[0], kernel_size=(3,3), stride=1,
                               padding=1, bias=False, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        # 1st block
        self.block1 = BayesianNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = BayesianNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = BayesianNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = BayesianLinearLayer(nChannels[3], n_classes, pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

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

        mean_log_probs = logmeanexp(log_probs, dim=0) # dim: (batch_size, n_classes),  transform to probabilities and take mean over samples
        mean_predictions = np.argmax(mean_log_probs, axis=1) # dim: (batch_size)

        return mean_predictions, mean_log_probs
    
    def compute_log_prior(self):
        model_log_prior = 0.0
        for layer in [self.conv1, self.block1, self.block2, self.block3, self.fc]:
            # layer can either be BayesianConvLayer or nn.Sequential() containing 4 Bayesian Blocks
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_prior += layer.log_prior
            elif isinstance(layer, BayesianNetworkBlock):
                for module in layer.modules():
                    if isinstance(module, BayesianLinearLayer) or isinstance(module, BayesianConvLayer):
                        model_log_prior += module.log_prior
        return model_log_prior
    
    def compute_log_variational_posterior(self):
        model_log_variational_posterior = 0.0
        for layer in [self.conv1, self.block1, self.block2, self.block3, self.fc]:
            # layer can either be BayesianConvLayer or nn.Sequential() containing 4 Bayesian Blocks
            if isinstance(layer, BayesianLinearLayer) or isinstance(layer, BayesianConvLayer):
                model_log_variational_posterior += layer.log_variational_posterior
            elif isinstance(layer, BayesianBasicBlock):
                for module in layer.modules():
                    if isinstance(module, BayesianLinearLayer) or isinstance(module, BayesianConvLayer):
                        model_log_variational_posterior += module.log_variational_posterior 
        return model_log_variational_posterior
    
    def compute_NLL(self, pred, target):
        loss_fn = torch.nn.NLLLoss(reduction='mean')
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

        loss = (weight*(log_variational_posterior - log_prior)) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, pred
    
