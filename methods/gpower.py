# Module to estimatimate Sparse Principal Components Analysis (SPCA):
# GPower: Generalized power method. Using cardinality (L0-norm) as a sparse penalty
# Author: RI. Guerra-Urzola & S. Park
# From: (Add journal reference)
# Date: 2024-11-22
# License: MIT License

# Importing required libraries

import numpy as np

def gpower(X, w0=None, alpha=0.1, maxiter=100, tol=1e-6):
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    _ ,p = X.shape
    norm_a_i = np.linalg.norm(X, axis=0)
    max_norm_a_i = np.max(norm_a_i)
    i_max = np.argmax(norm_a_i)
    if w0 is None:
        z = X[:, i_max] / norm_a_i[i_max]
        w0 = X.T @ z
    else:
        z =  X @ w0 / np.linalg.norm(X @ w0)
    iter = 0
    while iter < maxiter:
        Xz = X.T @ z
        threshold = np.maximum(Xz**2 - alpha * max_norm_a_i**2, 0)
        # Xz = Xz[threshold != 0] and 0 otherwise
        Xz = Xz * (threshold != 0) + 0 * (threshold == 0)
        if np.all(threshold == 0):
            w = np.zeros(p)
            break # Stopping criterion based on all zero weights
        
        w = Xz / np.linalg.norm(Xz)
    
        if np.linalg.norm(w - w0) < tol:
            break  # Stopping criterion based on tolerance
        w0 = w
        z = X @ w / np.linalg.norm(X @ w)  # Update weights
        iter += 1

    return w

