import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, RandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np

def load_cifar10(data_path: str, data_aug = True):
    # normalisation values from https://github.com/kuangliu/pytorch-cifar/issues/19
    # transform  = transforms.Compose(
    #     [transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # generator = torch.Generator().manual_seed(1871)
    # CIFAR_traindata = torchvision.datasets.CIFAR10(root=data_path, train=True, transform = transform, download=True)
    # CIFAR_train, CIFAR_val = torch.utils.data.random_split(CIFAR_traindata, [int(len(CIFAR_traindata)*0.9), int(len(CIFAR_traindata)*0.1)], generator=generator)

    # CIFAR_test = torchvision.datasets.CIFAR10(root=data_path, train=False, transform = transform, download=True)
    # return CIFAR_train, CIFAR_val, CIFAR_test
    
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    
    if data_aug == True:
        # Transforms from https://github.com/xternalz/WideResNet-pytorch/blob/master/train.py
        transform_train = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]
    )
    else:
        transform_train = transforms.Compose(
        [transforms.ToTensor(),
         normalize])

    transform_test  = transforms.Compose(
        [transforms.ToTensor(),
         normalize])
    
    generator = torch.Generator().manual_seed(1871)
    CIFAR_traindata = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
    len_train = int(len(CIFAR_traindata)*0.9)
    len_val = int(len(CIFAR_traindata)*0.1)
    randperm = torch.randperm(50000, generator=generator)

    CIFAR_train_x = CIFAR_traindata.data[randperm[:len_train]]
    CIFAR_train_y = np.array(CIFAR_traindata.targets)[randperm[:len_train]]
    CIFAR_val_x = CIFAR_traindata.data[randperm[len_train:]]
    CIFAR_val_y = np.array(CIFAR_traindata.targets)[randperm[len_train:]]
    CIFAR_train = CIFAR10(CIFAR_train_x, CIFAR_train_y, transform_train)
    CIFAR_val = CIFAR10(CIFAR_val_x, CIFAR_val_y, transform_test)

    CIFAR_test = torchvision.datasets.CIFAR10(root=data_path, train=False, transform = transform_test, download=True)
    return CIFAR_train, CIFAR_val, CIFAR_test

class CIFAR10(Dataset):
    def __init__(self, data, targets, transform):
        self.x = data
        self.y = targets
        self.transform=transform

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        y = self.y[idx]
        return x, y
    
    def __len__(self):
        return len(self.x)

class CIFAR10C(Dataset):
    def __init__(self, data_path, c_type, severity=1):
        data = np.load(data_path + c_type + '.npy')
        lb = 10000*severity - 10000
        ub = 10000*severity
        self.x = data[lb:ub]

        #compute normalisation:
        means = self.x.mean(axis=(0,1,2))/255
        stds = self.x.std(axis=(0,1,2))/255

        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))])
        self.transform  = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.y = np.load(data_path + 'labels.npy')

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        y = self.y[idx]
        return x, y
    
    def __len__(self):
        return len(self.x)

    

def load_CIFAR10C(data_path: str, type: str, severity = 1):
    '''
    Inputs:
    - data_path: the path to the cifar10-C data
    - type: the type of corruption to use

    Output:
    - the cifar10-C dataset, used for testing
    '''
    
    CIFAR_test = CIFAR10C(data_path, type, severity)

    return CIFAR_test



#Collate functions
def C_train_collate_fn(batch, M, batch_repetition=1):
    """Collate function for training MIMO on CIFAR classification"""
    
    x, y = zip(*batch)

    x, y = torch.stack(list(x)), torch.tensor(y)

    if batch_repetition > 1:
        x = x.repeat(batch_repetition, 1, 1, 1)
        y = y.repeat(batch_repetition)
    
        indices = torch.randperm(x.size(0))
        x = x[indices]
        y = y[indices]

    x = torch.cat(torch.chunk(x, M, dim=0), dim=1)
    y = torch.stack(torch.chunk(y, M, dim=0), dim=1)
    return x, y

def C_test_collate_fn(batch, M):
    """Collate function for testing MIMO on CIFAR classification"""
    
    x, y = zip(*batch)
    x, y = torch.stack(list(x)), torch.tensor(y)
    x = x.repeat(1, M, 1, 1)
    y = y[:,None].repeat(1,M)
    
    return x, y

def C_Naive_train_collate_fn(batch, M):
    """Collate function for training Naive multiheaded on CIFAR classification"""

    x, y = zip(*batch)

    x, y = torch.stack(list(x)), torch.tensor(y)
    y = y[:,None].repeat(1,M)

    return x, y

def C_Naive_test_collate_fn(batch, M):
    """Collate function for testing Naive multiheaded on CIFAR classsification"""

    x, y = zip(*batch)
    x, y = torch.stack(list(x)), torch.tensor(y)
    y = y[:,None].repeat(1,M)

    return x, y


if __name__ == '__main__':

    CIFAR_train, CIFAR_val, CIFAR_test = load_cifar10('data')
    inv_transform = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], std=[1/0.247, 1/0.243, 1/0.261])
    label_dict = {0: "airplane", 
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"}

    # plot some images
    fig, ax = plt.subplots(2,5, figsize=(18,6))
    fig.tight_layout()
    ax = ax.ravel()

    seen_labels = []
    i = 0
    for x,y in CIFAR_train:
        if y not in seen_labels:
            seen_labels.append(y)
            img = inv_transform(x) # unnormalise
            ax[i].imshow(img.permute(1,2,0))
            ax[i].set_title(f'Label: {label_dict[y]}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            i += 1

    print(os.getcwd())
    plt.savefig(r"reports/figures/CIFAR10example.png")
    plt.show()