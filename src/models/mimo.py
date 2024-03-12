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
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128):
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
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.model = torch.nn.Sequential(
            nn.Linear(self.n_subnetworks, hidden_units),
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
    def __init__(self, n_subnetworks):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 3
        self.channels1 = 32
        self.channels2 = 64

        self.conv = torch.nn.Sequential(
            nn.Conv2d(self.in_channels*self.n_subnetworks, self.channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels1, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels2, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.output = torch.nn.Sequential(
            nn.Linear(self.channels2 * 32 * 32, 128), # dim: self.channels2 x width x height
            nn.ReLU(),
            nn.Linear(128, self.n_subnetworks*10)
        )

    def forward(self, x):

        x = self.conv(x)
        # reshape to fit into linear layer
        x = x.reshape(x.size(0), -1)
        x = self.output(x)

        # reshape to batch_size x M x 10
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        x = nn.LogSoftmax(dim=2)(x)

        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(x, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(x, dim=1).argmax(dim=1) # dim : batch_size
        
        x = x.permute(1,0,2)

        return x, output, individual_outputs
    
class C_NaiveNetwork(nn.Module):
    def __init__(self, n_subnetworks):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 3
        self.channels1 = 32
        self.channels2 = 64

        self.conv = torch.nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels1, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels2, self.channels2, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.output = torch.nn.Sequential(
            nn.Linear(self.channels2 * 32 * 32, 128), # dim: self.channels2 x width x height
            nn.ReLU(),
            nn.Linear(128, self.n_subnetworks*10)
        )

    def forward(self, x):

        x = self.conv(x)
        # reshape to fit into linear layer
        x = x.reshape(x.size(0), -1)
        x = self.output(x)

        # reshape to batch_size x M x 10
        x = x.reshape(x.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        x = nn.LogSoftmax(dim=2)(x)
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(x, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(x, dim=1).argmax(dim=1) # dim : batch_size

        x = x.permute(1,0,2)

        return x, output, individual_outputs
    
class VarNaiveNetwork(nn.Module):
    def __init__(self, n_subnetworks, hidden_units=32, hidden_units2=128):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.channels1 = 16
        self.channels2 = 32
        self.model = torch.nn.Sequential(
            nn.Linear(1, hidden_units),
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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
    Wide ResNet model for MIMO classification. 
    """
    def __init__(self, n_subnetworks, depth, widen_factor, dropout_rate, num_classes=10):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 16
        
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = self.conv3x3(3*self.n_subnetworks, nStages[0])
        self.layer1 = self._wide_layer(BasicWideBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(BasicWideBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(BasicWideBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes*self.n_subnetworks)

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
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # reshape to batch_size x M x 10
        out = out.reshape(out.size(0), self.n_subnetworks, -1)
        # Log-softmax over the last dimension (because we are using NLL loss)
        out = nn.LogSoftmax(dim=2)(out)
        
        # get individual outputs 
        # during training, we want each subnetwork to to clasify their corresponding inputs
        individual_outputs = torch.argmax(out, dim=2) # dim : batch_size x M
        
        # get ensemble output
        # during inference, we mean the softmax probabilities over all M subnetworks and then take the argmax
        output = torch.mean(x, dim=1).argmax(dim=1) # dim : batch_size

        out = out.permute(1,0,2)

        return out, individual_outputs, output

if __name__ == '__main__':
    model = MIMOWideResnet(n_subnetworks=2, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10) 
    
