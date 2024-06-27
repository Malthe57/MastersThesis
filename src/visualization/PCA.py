import numpy as np

def PCA(X, n_components=2, return_vectors=True):

    X_tilde = X - np.mean(X,axis=0) #Check om rigtig axis
    # X_tilde = X_tilde/np.std(X, axis=0) #Igen check axis

    #Compute SVD
    sigmas, V = np.linalg.eig(X_tilde.T@X_tilde)
    sigmas2, U = np.linalg.eig(X_tilde@X_tilde.T)
    cov = np.cov(X_tilde.T)
    eigenvalues, eigenvector = np.linalg.eigh(cov)
    total_var = np.sum(eigenvalues)
    sort_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_idx]
    for i in range(n_components):
        print(f'Variance explained by component {i}: {sorted_eigenvalues[i]/total_var}')
    sorted_vectors = eigenvector[:,sort_idx]
    if return_vectors:
        return sorted_vectors

    else:
        X_pca = np.matmul(X_tilde, sorted_vectors[:,:n_components])

    return X_pca