# Module to estimatimate Sparse Principal Components Analysis (SPCA):
# CP-SPCA: our proposed algorithm. Using cardinality (L0-norm) as a sparse penalty
# Author: RI. Guerra-Urzola & S. Park
# Date: 2024-11-15
# License: MIT License

# Importing required libraries
import numpy as np

def cp_spca(X, w0=None, alpha=1.0, maxiter=100, tol=1e-6):
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    
    if alpha == 0:  # if alpha is zero, return PCA solution
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt[0, :]
    
    n_samples, n_features = X.shape
    Sig = (X.T @ X) / n_samples  # sample covariance matrix
    Sig += 0.001 * np.eye(n_features)  # add small value to diagonal
    
    if w0 is None:  # Initialize weights if not provided
        w0 = np.random.randn(n_features)
        w0 /= np.linalg.norm(w0)  # Normalize w0 to have norm 1
    
    alpha /= 2  # alpha is divided by 2 to match the solution in the paper
    
    iter = 0
    while iter < maxiter:
        Sigw = Sig @ w0
        w = np.where(Sigw >= alpha, Sigw, 0)  # Apply thresholding
        
        if np.all(w == 0):
            break  # Stopping criterion based on all zero weights
        
        w /= np.linalg.norm(w)  # Normalize weights
        
        if np.linalg.norm(w - w0) < tol:
            break  # Stopping criterion based on tolerance
        
        w0 = w  # Update weights
        iter += 1
    
    return w


