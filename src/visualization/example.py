import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
# np.random.seed(42)
x_train = np.sort(sorted(np.random.uniform(-5, 5, size=(100, 1)).astype(np.float32)))
y_train = np.sin(x_train) + np.random.normal(0, 0.1, size=(100, 1)).astype(np.float32)


# Remove a portion of the training data
num_points_to_remove = 50
indices_to_remove = np.array([list(range(0,25)) + list(range(40,70)) + list(range(85,100))])
x_removed = x_train[indices_to_remove]
y_removed = y_train[indices_to_remove]
x_train = np.delete(x_train, indices_to_remove, axis=0)
y_train = np.delete(y_train, indices_to_remove, axis=0)

x_test = np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32)
y_true = np.sin(x_test)

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
def train_model(x_train, y_train, num_samples):
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
        predictions = train_model(x_train, y_train, num_samples)
        plot_results(x_test, y_true, predictions)
    plt.legend(['Model predictions', 'Training data'])
    plt.grid()
    plt.tight_layout()
    plt.savefig('reports/figures/NN_preds.png')
    plt.show()
