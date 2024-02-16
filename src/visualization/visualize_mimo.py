import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss(losses, val_losses, model_name="MIMO"):

    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax.set_title(f"{model_name} Losses")

    ax[0].plot(losses, label='Train loss')
    ax[0].set_title('Train loss')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')

    ax[1].plot(val_losses, label='Validation loss', color='orange')
    ax[1].set_title('Validation loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')
    plt.savefig(f"reports/figures/{model_name}_losses.png")  
    plt.show()


def plot_weight_distribution(MIMO_model, Naive_model, mode = 'Classification'):
# weight distribution
    weights_mimo = []
    for param in MIMO_model.parameters():
        weights_mimo.extend(param.flatten().cpu().detach().numpy())

    weights_naive = []
    for param in Naive_model.parameters():
        weights_naive.extend(param.flatten().cpu().detach().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(weights_mimo, bins=50, alpha=0.5, label='MIMO_model')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Weight Distribution - Classification MIMO Model')

    ax2.hist(weights_naive, bins=50, alpha=0.5, label='naive_model')
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Weight Distribution - Classification Naive Model')

    plt.tight_layout()
    plt.savefig(f"images/{model_name}_{mode}_weights")
    plt.show()

def reliability_plot(accuracy, confidence):
        #Code for generating reliability diagram:
    bins_range = np.arange(0, 1, 0.1)
    conf_step_height = np.zeros(10)
    acc_step_height = np.zeros(10)
    for i in range(len(bins_range)-1):
        loc = np.where(confidence>=bins_range[i] and confidence<bins_range[i+1])
        conf_step_height[i] = np.mean(confidence[loc])
        acc_step_height[i] = np.mean(accuracy[loc])

    naive_conf_step_height = np.zeros(10)
    naive_acc_step_height = np.zeros(10)
    for i in range(len(bins_range)-1):
        loc = np.where(confidence>=bins_range[i] and confidence<bins_range[i+1])
        naive_conf_step_height[i] = np.mean(confidence[loc])
        naive_acc_step_height[i] = np.mean(accuracy[loc])



    ax, fig = plt.subplots(2,1, sharey=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax[0].stairs(conf_step_height, bins_range, hatch="//")
    ax[0].stairs(acc_step_height, bins_range, fill = True)
    ax[0].set_title("Reliability Plot MIMO")

    ax[1].stairs(conf_step_height, bins_range, hatch="//", legend='Gap')
    ax[1].stairs(acc_step_height, bins_range, fill = True, legend="Outputs")
    ax[1].set_title("Reliability Plot Naive Multiheaded")
    plt.tight_layout()
    plt.savefig("images/confidence_plots.png")
    plt.show()

def function_space_plots(model_name):
    checkpoint_list = torch.load(f'../../models/{model_name}.pt')
    checkpoint_list = torch.stack(checkpoint_list[:,:5,:]).flatten()
    tSNE = TSNE(checkpoint_list.shape)
    val_checkpoint_list2d = tSNE.fit_transform(checkpoint_list)
    plt.plot(checkpoint_list)


if __name__ == '__main__':
    model_name = "MIMO"
    model = torch.load('../../models/f{model_name}.pt')

