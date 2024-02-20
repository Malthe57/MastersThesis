import torch.nn as nn
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

        # mean mu and sigma 
        mu = torch.mean(mus, dim=1)
        sigma = torch.mean(sigmas, dim=1)
        
        return mu, sigma, mus, sigmas
    

class C_MIMONetwork(nn.Module):
    def __init__(self, n_subnetworks):
        super().__init__()
        self.n_subnetworks = n_subnetworks
        self.in_channels = 3
        self.channels1 = 16
        self.channels2 = 32

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
            nn.Dropout(p=0.1),
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
        self.channels1 = 16
        self.channels2 = 32

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
            nn.Dropout(p=0.1),
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