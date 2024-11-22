import numpy as np
from methods.multivariate_spca import MultiSpca

def test_multi_spca():
    # Test 1: Basic functionality with n_comp = 1
    np.random.seed(0)
    X = np.random.randn(100, 5)
    n_comp = 1
    alpha = 0.5
    components = MultiSpca(X, n_comp=n_comp, Alpha=alpha)
    assert components.shape == (5,n_comp), "Test 1 failed: Incorrect output shape for n_comp = 1."
    assert np.linalg.norm(components[0]) - 1 < 1e-6, "Test 1 failed: Output weights are not normalized."
    
    # Test 2: Multiple components with n_comp = 3
    n_comp = 3
    alpha = 0.5
    components = MultiSpca(X, n_comp=n_comp, Alpha=alpha)
    assert components.shape == (5,n_comp), "Test 2 failed: Incorrect output shape for n_comp = 3."
    
    # Test 3: Test with Alpha as a vector
    Alpha = [0.5, 0.3, 0.1]
    components = MultiSpca(X, n_comp=n_comp, Alpha=Alpha)
    assert components.shape == (5,n_comp), "Test 3 failed: Incorrect output shape for Alpha as a vector."
    
    # Test 4: Test with unknown method
    try:
        MultiSpca(X, n_comp=n_comp, Alpha=alpha, method="unknown")
    except ValueError as e:
        print(f"Actual error message: '{str(e)}'")
        assert str(e) == "Unknown method: unknown. Choose from ['cp_pca', 'gpower'].", "Test 4 failed: Incorrect error message for unknown method."
    else:
        assert False, "Test 4 failed: ValueError not raised for unknown method."
    
    # Test 5: Test with gpower method
    components = MultiSpca(X, n_comp=n_comp, Alpha=alpha, method="gpower")
    assert components.shape == (5,n_comp), "Test 5 failed: Incorrect output shape for gpower method."
    


    print("All tests passed!")

if __name__ == "__main__":
    test_multi_spca()
