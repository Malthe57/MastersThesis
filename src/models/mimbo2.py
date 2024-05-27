import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd() + '/src/')
from utils.utils import logmeanexp
from models.bnn import ScaleMixturePrior, Gaussian, BayesianLinearLayer, BayesianConvLayer
from models.bnn2 import BayesianBasicBlock, BayesianNetworkBlock

class MIMBOWideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.0, num_classes=10, n_subnetworks=1):
        super(MIMBOWideResNet, self).__init__()
        print(f"Initializing MIMBO WideResNet with {n_subnetworks} subnetworks")
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BayesianBasicBlock
        # 1st conv before any network block
        self.conv1 = BayesianConvLayer(3*n_subnetworks, nChannels[0], kernel_size=(3,3), stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = BayesianNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = BayesianNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = BayesianNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = BayesianLinearLayer(nChannels[3], num_classes*n_subnetworks)
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

        # reshape to batch_size x M x n_classes
        out = out.reshape(out.size(0), self.n_subnetworks, -1)
        out = out.permute(0, 2, 1) # dim : batch_size x n_classes x M
        # Log-softmax over the last dimension (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=1)(out) # dim : batch_size x n_classes x M
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=1) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = logmeanexp(log_probs, dim=2).argmax(dim=1) # dim : batch_size

        return log_probs, output, individual_outputs
    
    def inference(self, x, sample=True, n_samples=1, n_classes=10):
        # log_probs : (n_samples, batch_size, n_classes, n_subnetworks)
        log_probs_matrix = np.zeros((n_samples, x.size(0),  n_classes, self.n_subnetworks))

        for i in range(n_samples):
            output, individual_outputs, log_probs = self.forward(x, sample)
            log_probs_matrix[i] = log_probs.cpu().detach().numpy()

        mean_subnetwork_log_probs = logmeanexp(log_probs_matrix, dim=3)  # mean over n_subnetworks, dim : n_samples x batch_size x n_classes
        mean_log_probs = logmeanexp(mean_subnetwork_log_probs, dim=0) # mean over samples, dim : batch_size x n_classes
        mean_predictions = np.argmax(mean_log_probs, axis=1) # argmax over n_classes, dim : batch_size

        return mean_predictions, mean_subnetwork_log_probs, mean_log_probs
    
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
    
    def compute_NLL(self, log_probs, target, val=False):
        loss_fn = torch.nn.NLLLoss(reduction='mean')
        if val:
            # mean over log_probs over n_subnetworks dimension
            NLL = loss_fn(logmeanexp(log_probs), target[:,0])

        else:
            NLL = loss_fn(log_probs, target)

        return NLL
    
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def compute_ELBO(self, input, target, weight, n_samples=1, val = False):
        log_priors = torch.zeros(n_samples) 
        log_variational_posteriors = torch.zeros(n_samples) 
        NLLs = torch.zeros(n_samples) 

        for i in range(n_samples):
            output, individual_outputs, probs = self.forward(input, sample=True)
            log_priors[i] = self.compute_log_prior()
            log_variational_posteriors[i] = self.compute_log_variational_posterior()
            NLLs[i] = self.compute_NLL(probs, target, val=val)
            if val:
                pred = output
            else:
                pred = individual_outputs

        log_prior = log_priors.mean(0)
        log_variational_posterior = log_variational_posteriors.mean(0)
        NLL = NLLs.mean(0)

        loss = (weight * (log_variational_posterior - log_prior)) + NLL
 
        return loss, log_prior, log_variational_posterior, NLL, probs, pred
    
if __name__ == '__main__':
    net = MIMBOWideResNet(28, 10, 10, 0.3, 3)
    x = torch.randn(1, 3, 32, 32)
    log_probs, output, individual_outputs = net(x)
    net.compute_log_prior()
    net.compute_log_variational_posterior()