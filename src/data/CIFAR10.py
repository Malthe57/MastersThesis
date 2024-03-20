import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import os

def load_cifar10(data_path: str):
    # normalisation values from https://github.com/kuangliu/pytorch-cifar/issues/19
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


if __name__ == '__main__':

    CIFAR_train, CIFAR_val, CIFAR_test = load_cifar('data')
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