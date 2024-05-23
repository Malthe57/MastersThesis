import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

def load_cifar100(data_path: str):
    # values taken from https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    transform = transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    generator = torch.Generator().manual_seed(1871)
    CIFAR_traindata = torchvision.datasets.CIFAR100(root=data_path, train=True, transform = transform, download=True)
    CIFAR_train, CIFAR_val = torch.utils.data.random_split(CIFAR_traindata, [int(len(CIFAR_traindata)*0.9), int(len(CIFAR_traindata)*0.1)], generator=generator)

    CIFAR_test = torchvision.datasets.CIFAR100(root=data_path, train=False, transform = transform, download=True)
    return CIFAR_train, CIFAR_val, CIFAR_test

class CIFAR100C(Dataset):
    def __init__(self, data_path, c_type, severity=1):
        data = np.load(data_path + c_type + '.npy')
        lb = 10000*severity - 10000
        ub = 10000*severity
        self.x = data[lb:ub]

        #compute normalisation:
        means = self.x.mean(axis=(0,1,2))/255
        stds = self.x.std(axis=(0,1,2))/255

        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))])
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])
        
        self.y = np.load(data_path + 'labels.npy')

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        y = self.y[idx]
        return x, y
    
    def __len__(self):
        return len(self.x)
    
def load_CIFAR100C(data_path: str, type: str, severity = 1):
    '''
    Inputs:
    - data_path: the path to the cifar100-C data
    - type: the type of corruption to use

    Output:
    - the cifar100-C dataset, used for testing
    '''
    
    CIFAR_test = CIFAR100C(data_path, type, severity)

    return CIFAR_test

if __name__ == '__main__':
    CIFAR_train, CIFAR_val, CIFAR_test = load_cifar100('data')
    inv_transform = transforms.Normalize(mean=[-0.5071/0.267, -0.4865/0.256, -0.4409/0.276], std=[1/0.267, 1/0.256, 1/0.276])
    label_dict = {36: "hamster", 
                28: "cups",
                29: "dinosaur",
                2: "baby",
                53: "oranges",
                73: "shark",
                1: "aquarium fish",
                88: "tiger",
                99: "worm",
                95: "whale"}

    # plot some images
    fig, ax = plt.subplots(2,5, figsize=(18,6))
    fig.tight_layout()
    ax = ax.ravel()

    seen_labels = []
    i = 0
    for x,y in CIFAR_train:
        if y not in seen_labels and y in label_dict.keys():
            seen_labels.append(y)
            img = inv_transform(x) # unnormalise
            ax[i].imshow(img.permute(1,2,0))
            ax[i].set_title(f'Label: {label_dict[y]}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            i += 1

    plt.savefig(r"reports/figures/CIFAR100example.png")
    plt.show()
