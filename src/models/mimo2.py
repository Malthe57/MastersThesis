import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
sys.path.append(os.getcwd() + '/src/')
from utils.utils import logmeanexp

class BasicWideBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0):
        super(BasicWideBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):

        if not self.equalInOut: # if in_planes != out_planes
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class MIMOWideResnet(nn.Module):
    def __init__(self, n_subnetworks, depth, widen_factor, dropout_rate=0.0, n_classes=10):
        super(MIMOWideResnet, self).__init__()
        self.n_subnetworks = n_subnetworks
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6


        block = BasicWideBlock
        # 1st conv before any network block
        self.conv1_layer = nn.Conv2d(3*n_subnetworks, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], n_classes*n_subnetworks)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1_layer(x)
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


if __name__ == '__main__':
    pass