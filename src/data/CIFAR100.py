import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def load_cifar100(data_path: str):
    # values taken from https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    transform = transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])

    CIFAR_traindata = torchvision.datasets.CIFAR100(root=data_path, train=True, transform = transform, download=True)
    CIFAR_train, CIFAR_val = torch.utils.data.random_split(CIFAR_traindata, [int(len(CIFAR_traindata)*0.9), int(len(CIFAR_traindata)*0.1)])

    CIFAR_test = torchvision.datasets.CIFAR100(root=data_path, train=False, transform = transform, download=True)
    return CIFAR_train, CIFAR_val, CIFAR_test


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
