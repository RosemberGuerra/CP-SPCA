import numpy as np
from methods.gpower import gpower

def test_gpower():
    # Test 1: Basic functionality with alpha = 0.2
    np.random.seed(0)
    X = np.random.randn(100,5)
    alpha = 0.2
    w = gpower(X, alpha=alpha)
    assert w.shape == (5,), "Test 1 failed: Incorrect output shape."
    assert np.linalg.norm(w) - 1 < 1e-6, "Test 1 failed: Output weights are not normalized."
    
    # Test 2: Test with provided initial weights (w0)
    w0 = np.random.randn(5)
    w0 /= np.linalg.norm(w0)
    w = gpower(X, w0=w0, alpha=alpha)
    assert w.shape == (5,), "Test 2 failed: Incorrect output shape when providing initial weights."
    assert np.linalg.norm(w) - 1 < 1e-6, "Test 2 failed: Output weights are not normalized when providing initial weights."
    
    # Test 3: Test with large alpha (to check all zero weights case)
    alpha = 10.0
    w = gpower(X, alpha=alpha)
    assert np.allclose(w, 0), "Test 3 failed: Output weights should be all zeros for large alpha."
    
    # Test 4: Test with negative alpha (to check exception handling)
    try:
        gpower(X, alpha=-0.5)
    except ValueError as e:
        assert str(e) == "alpha must be non-negative.", "Test 4 failed: Incorrect error message for negative alpha."
    else:
        assert False, "Test 4 failed: ValueError not raised for negative alpha."
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gpower()