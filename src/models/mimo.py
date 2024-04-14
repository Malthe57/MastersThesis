import torch.nn as nn
import torch.nn.functional as F
import torch

class MIMONetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.model = torch.nn.Sequential(
            nn.Linear(self.n_subnetworks, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units2),
            nn.ReLU(),
            nn.Linear(hidden_units2, self.n_subnetworks)
        )


    def forward(self, x):
        individual_outputs = self.model(x)
        output = torch.mean(individual_outputs, dim=1)
        return output, individual_outputs
    
class NaiveNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128, input_dim = 1):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.model = torch.nn.Sequential(
            nn.Linear(1, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units2),
            nn.ReLU(),
            nn.Linear(hidden_units2, self.n_subnetworks)
        )

    def forward(self, x):
        individual_outputs = self.model(x)
        output = torch.mean(individual_outputs, dim=1)
        return output, individual_outputs

class VarMIMONetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128, input_dim=1):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim*self.n_subnetworks, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units2),
            nn.ReLU(),
            nn.Linear(hidden_units2, self.n_subnetworks*2)
        )

    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x):
        individual_outputs = self.model(x)
        # mus and sigmas for each subnetwork
        mus = individual_outputs[:,:self.n_subnetworks]
        sigmas = self.get_sigma(individual_outputs[:,self.n_subnetworks:])

        # compute mu and sigma for mixture model with M gaussian
        # https://stats.stackexchange.com/a/445232
        mu = torch.mean(mus, dim=1)
        sigma = (torch.mean((mus.pow(2) + sigmas.pow(2)), dim=1) - mu.pow(2)).sqrt()
        
        return mu, sigma, mus, sigmas
    

class C_MIMONetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1=128, channels1=64, channels2=128, channels3=256, n_classes=10):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.n_classes = n_classes
        self.in_channels = 3
        self.hidden_units1 = hidden_units1
        self.channels1 = channels1
        self.channels2 = channels2
        self.channels3 = channels3

        self.conv = torch.nn.Sequential(
            nn.Conv2d(self.in_channels*self.n_subnetworks, self.channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels1, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels2, self.channels3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels3, self.channels3, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.output = torch.nn.Sequential(
            nn.Linear(self.channels3* 32 * 32, self.hidden_units1), # dim: self.channels2 x width x height
            nn.ReLU(),
            nn.Linear(self.hidden_units1, self.n_subnetworks*self.n_classes)
        )

    def forward(self, x):

        x = self.conv(x)
        # reshape to fit into linear layer
        x = x.reshape(x.size(0), -1)
        x = self.output(x)

        # reshape to batch_size x M x n_classes
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=2)(x) # dim : batch_size x M x n_classes

        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size
        
        log_probs = log_probs.permute(1,0,2) # dim : M x batch_size x n_classes

        return log_probs, output, individual_outputs
    
class C_NaiveNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units1=128, channels1=64, channels2=128, channels3=256, n_classes=10):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.n_classes = n_classes
        self.in_channels = 3
        self.hidden_units1 = hidden_units1
        self.channels1 = channels1
        self.channels2 = channels2
        self.channels3 = channels3

        self.conv = torch.nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels1, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels2, self.channels3, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels3, self.channels3, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.output = torch.nn.Sequential(
            nn.Linear(self.channels3 * 32 * 32, self.hidden_units1), # dim: self.channels2 x width x height
            nn.ReLU(),
            nn.Linear(self.hidden_units1, self.n_subnetworks*self.n_classes)
        )

    def forward(self, x):

        x = self.conv(x)
        # reshape to fit into linear layer
        x = x.reshape(x.size(0), -1)
        x = self.output(x)

        # reshape to batch_size x M x n_classes
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=2)(x)
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size

        log_probs = log_probs.permute(1,0,2)

        return log_probs, output, individual_outputs
    
class VarNaiveNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128, input_dim = 1):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.channels1 = 16
        self.channels2 = 32
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units2),
            nn.ReLU(),
            nn.Linear(hidden_units2, self.n_subnetworks*2)
            )
    def get_sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x):
        individual_outputs = self.model(x)
        # mus and sigmas for each subnetwork
        mus = individual_outputs[:,:self.n_subnetworks]
        sigmas = self.get_sigma(individual_outputs[:,self.n_subnetworks:])

        # compute mu and sigma for mixture model with M gaussian
        # https://stats.stackexchange.com/a/445232
        mu = torch.mean(mus, dim=1)
        sigma = (torch.mean((mus.pow(2) + sigmas.pow(2)), dim=1) - mu.pow(2)).sqrt()

        return mu, sigma, mus, sigmas
    

class BasicWideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.3, stride=1):
        """"
        Basic wide block used in Wide ResNet. It consists of two convolutional layers (with dropout in between) 
        and a skip connection. 

        Inputs:
        - in_channels: number of input channels
        - out_channels: number of output channels
        - stride: stride of the first convolutional layer
        - p: dropout probability

        Returns:
        - out: output tensor
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=p)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        # skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )

    def forward (self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.skip(x)

        return out
    

class MIMOWideResnet(nn.Module):
    """
    Wide ResNet model for MIMO classification. Code adapted from https://github.com/meliketoy/wide-resnet.pytorch/tree/master
    """
    def __init__(self, n_subnetworks, depth, widen_factor, dropout_rate, n_classes=10):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 16
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.layer1 = self.conv3x3(3*self.n_subnetworks, nStages[0])
        self.layer2 = self._wide_layer(BasicWideBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer3 = self._wide_layer(BasicWideBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer4 = self._wide_layer(BasicWideBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], n_classes*self.n_subnetworks)

    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def _wide_layer(self, block, out_channels, num_blocks, p, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, p, stride))
            self.in_channels = out_channels


        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # reshape to batch_size x M x n_classes
        out = out.reshape(out.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=2)(out) # dim : batch_size x M x n_classes
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size

        log_probs = log_probs.permute(1,0,2) # dim : M x batch_size x n_classes

        return log_probs, output, individual_outputs
    
class NaiveWideResnet(nn.Module):
    """
    Wide ResNet model for naive classification. Code adapted from https://github.com/meliketoy/wide-resnet.pytorch/tree/master
    """
    def __init__(self, n_subnetworks, depth, widen_factor, dropout_rate, n_classes=10):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 16
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.layer1 = self.conv3x3(3, nStages[0])
        self.layer2 = self._wide_layer(BasicWideBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer3 = self._wide_layer(BasicWideBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer4 = self._wide_layer(BasicWideBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], n_classes*self.n_subnetworks)

    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def _wide_layer(self, block, out_channels, num_blocks, p, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, p, stride))
            self.in_channels = out_channels


        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # reshape to batch_size x M x n_classes
        out = out.reshape(out.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        log_probs = nn.LogSoftmax(dim=2)(out) # dim : batch_size x M x n_classes
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(log_probs, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(log_probs, dim=1).argmax(dim=1) # dim : batch_size

        log_probs = log_probs.permute(1,0,2) # dim : M x batch_size x n_classes

        return log_probs, output, individual_outputs

if __name__ == '__main__':
    model = MIMOWideResnet(n_subnetworks=2, depth=28, widen_factor=10, dropout_rate=0.3, n_classes=10) 
    
