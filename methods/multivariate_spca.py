from methods.cp_pca import cp_spca
#from gpower import gpower

import numpy as np

def MultiSpca(data, n_comp, Alpha, method = "cp_pca"):
    """
    Compute multiple principal components using deflation.
    Args:
        data (numpy.ndarray): Input data matrix (n_samples, n_features).
        n_components (int): Number of principal components to compute.
        W0 (numpy.ndarray): Initial weights for the first principal component (n_features,).
        Alpha (float): Regularization parameter for sparse PCA. Vector or scalar.
        method (str): Method to use for computing the first principal component: 'pcpca', 'gpower'.
    Returns:
        numpy.ndarray: Matrix of principal components (n_components, n_features).
    """
    # Check input Alpha is a vector or scalar
    if isinstance(Alpha, float):
        Alpha = [Alpha] * n_comp


     # Mapping method names to functions
    methods = {
        "cp_pca": cp_spca
        #"gpower": gpower
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}.")

    method_func = methods[method]
    components = []
    for i in range(n_comp):
        # Compute first principal component
        w = method_func(data, alpha=Alpha[i])
        components.append(w)

        # Deflate the data
        data = data -  (data @ w[:, np.newaxis]) @ w[np.newaxis, :]

    return np.array(components).T
