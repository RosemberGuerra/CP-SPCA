import numpy as np
from methods.pcpca import cp_spca

def test_cp_spca():
    # Test 1: Basic functionality with alpha = 0 (PCA solution)
    np.random.seed(0)
    X = np.random.randn(100, 5)
    w = cp_spca(X, alpha=0)
    assert w.shape == (5,), "Test 1 failed: Incorrect output shape for PCA solution."
    
    # Test 2: Basic functionality with alpha > 0
    alpha = 0.5
    w = cp_spca(X, alpha=alpha)
    assert w.shape == (5,), "Test 2 failed: Incorrect output shape for sparse PCA solution."
    assert np.linalg.norm(w) - 1 < 1e-6, "Test 2 failed: Output weights are not normalized."
    
    # Test 3: Test with provided initial weights (w0)
    w0 = np.random.randn(5)
    w0 /= np.linalg.norm(w0)
    w = cp_spca(X, w0=w0, alpha=alpha)
    assert w.shape == (5,), "Test 3 failed: Incorrect output shape when providing initial weights."
    assert np.linalg.norm(w) - 1 < 1e-6, "Test 3 failed: Output weights are not normalized when providing initial weights."
    
    # Test 4: Test with alpha large enough to yield all zero weights
    alpha = 10.0
    w = cp_spca(X, alpha=alpha)
    assert np.all(w == 0), "Test 4 failed: Output weights should be all zeros for large alpha."
    
    print("All tests passed!")

if __name__ == "__main__":
    test_cp_spca()
