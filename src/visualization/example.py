import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_multidim_data(N, lower, upper, std, dim=1, num_points_to_remove=0, projection_matrix=None, save_x_path=None):

    # create data
    x_1d = np.linspace(lower, upper, N)
    
    # noise std from ]-inf, 0.5] and noise 5*std from [0.5, inf[
    n1 = len(x_1d[x_1d>=1000])
    noise_range = np.linspace(0, 1, n1)
    noise1 = np.random.normal(0, 1, N-n1) * std
    noise2 = np.random.normal(0,1, n1) *(1 + noise_range * 4) * std
    noise = np.concatenate((noise1, noise2))

    # Regression data function
    y = x_1d + 0.3 * np.sin(2*np.pi * (x_1d + noise)) + 0.3 * np.sin(4 * np.pi * (x_1d + noise)) + noise

    # project to multidimensional space
    if dim > 1:
        x = np.dot(x_1d[:,None], projection_matrix)
    else: 
        x = x_1d
    
    if num_points_to_remove > 0:
        start = (len(x) // 3) - (num_points_to_remove//2)
        end = start + num_points_to_remove
        x_1d = np.delete(x_1d, np.s_[start:end], axis=0)
        x  = np.delete(x, np.s_[start:end], axis=0)
        y = np.delete(y, np.s_[start:end], axis=0)

    if save_x_path is not None:
        np.savez(f"{save_x_path}/x_1d.npz", x_1d=x_1d)

    return x_1d, y

# Generate some synthetic data
np.random.seed(42)
# x_train = np.sort(sorted(np.random.uniform(-5, 5, size=(100, 1)).astype(np.float32)))
# y_train = np.sin(x_train) + np.random.normal(0, 0.1, size=(100, 1)).astype(np.float32)

# Remove a portion of the training data
# num_points_to_remove = 50
# indices_to_remove = np.array([list(range(0,25)) + list(range(40,70)) + list(range(85,100))])
# x_removed = x_train[indices_to_remove]
# y_removed = y_train[indices_to_remove]
# x_train = np.delete(x_train, indices_to_remove, axis=0)
# y_train = np.delete(y_train, indices_to_remove, axis=0)

# x_test = np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32)
# y_true = np.sin(x_test)

dim = 1
num_points_to_remove = 200
projection_matrix = np.random.randn(1, dim)
# Generate train data
N_train = 1000
x_train, y_train = generate_multidim_data(N_train, lower=-0.25, upper=1, std=0.02, dim=dim, num_points_to_remove=num_points_to_remove, projection_matrix=projection_matrix)

# Generate validation data
N_val = 5000
x_test, y_test = generate_multidim_data(N_val, lower=-0.25, upper=1, std=0.02, dim=dim, projection_matrix=projection_matrix)

x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)


# Define Bayesian neural network model
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the Bayesian neural network
def train_model(x_train, y_train, x_test):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    hidden_dim = 50
    num_epochs = 500
    learning_rate = 0.01

    model = BayesianNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch.tensor(x_train))
        loss = criterion(outputs, torch.tensor(y_train))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Generate predictions from multiple sets of parameters
    predictions = model(torch.tensor(x_test)).detach().numpy()

    return predictions

# Plot the results
def plot_results(x_test, y_true, predictions):
    plt.scatter(x_train, y_train, color='black', label='Training data')

    for i, prediction in enumerate(predictions.T):
        if i == 0:
            plt.plot(x_test, prediction,  c='C0', alpha=0.3)
        else:
            plt.plot(x_test, prediction,  c='C0', alpha=0.3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural network predictions')
    

if __name__ == "__main__":

    num_samples = 10
    plt.figure(figsize=(10, 6))
    for i in range(50):
        torch.manual_seed(i*175)
        predictions = train_model(x_train[:,None], y_train[:,None], x_test[:,None])
        plot_results(x_test[:,None], y_test[:,None], predictions)
    plt.legend(['Training data', 'Model predictions'])
    plt.grid()
    plt.tight_layout()
    plt.savefig('reports/figures/NN_preds.png')
    plt.show()
