import torch
import torchvision
from torchvision.transforms import transforms

def load_cifar(data_path: str):
    transform = transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    CIFAR_traindata = torchvision.datasets.CIFAR10(root=data_path, train=True, transform = transform, download=True)
    CIFAR_train, CIFAR_val = torch.utils.data.random_split(CIFAR_traindata, [int(len(CIFAR_traindata)*0.9), int(len(CIFAR_traindata)*0.1)])

    CIFAR_test = torchvision.datasets.CIFAR10(root=data_path, train=False, transform = transform, download=True)
    return CIFAR_train, CIFAR_val, CIFAR_test

#Collate functions
def C_train_collate_fn(batch, M):
    """Collate function for training MIMO on CIFAR classification"""
    
    x, y = zip(*batch)

    x, y = torch.stack(list(x)), torch.tensor(y)
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