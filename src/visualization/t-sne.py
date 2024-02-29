import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from models.mimo import C_MIMONetwork
import torch
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

X1 = torch.load("notebooks/C_MIMO_2_checkpoints.pt").cpu().numpy().squeeze(0)
X2 = torch.load("notebooks/C_MIMO_3_checkpoints.pt").cpu().numpy().squeeze(0)
X3 = torch.load("notebooks/C_MIMO_4_checkpoints.pt").cpu().numpy().squeeze(0)

model1 = torch.load("notebooks/C_MIMO_2.pt")
model2 = torch.load("notebooks/C_MIMO_3.pt")
model3 = torch.load("notebooks/C_MIMO_4.pt")

theta1 = Params2Vec(model1.parameters())

pca = PCA(n_components=2)
components = pca.fit_transform(theta1.detach().numpy())

# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

# # plot X_embedded
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
# plt.show()

print("pass")