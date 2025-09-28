cp_spca <- function(X, w0 = NULL, alpha = 1.0, tao = 0.001 ,maxiter = 100, tol = 1e-6) {
  # X: the data set
  # w0: initial weights
  # alpha: level of sparsity 
  
  if (alpha < 0) stop("alpha must be non-negative.")
  
  # If alpha == 0, return PCA solution (first right singular vector)
  if (alpha == 0) {
    sv <- svd(X)
    return(list(w=  as.vector(sv$v[, 1]), iter = 0, time = 0))
  }
  
  X <- as.matrix(X)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Sample covariance with small ridge on the diagonal
  Sig <- crossprod(X) / n_samples
  Sig <- Sig + tao * diag(n_features)
  
  # Initialize w0 (unit-norm) if not provided
  if (is.null(w0)) {
    w0 <- rnorm(n_features)
    w0 <- w0 / sqrt(sum(w0^2))
  } else {
    w0 <- as.numeric(w0)
    if (length(w0) != n_features) stop("w0 length must match ncol(X).")
    nrm <- sqrt(sum(w0^2))
    if (nrm == 0) stop("w0 must be non-zero.")
    w0 <- w0 / nrm
  }
  
  # Match the paper's scaling
  alpha <- alpha / 2
  
  iter <- 0
  runningtime <- system.time({
  repeat {
    if (iter >= maxiter) break
    
    Sigw <- as.vector(Sig %*% w0)
    # Element-wise thresholding: keep entries >= alpha, else set to 0
    w <- ifelse(abs(Sigw) >= alpha, Sigw, 0)
    
    # Stop if all weights are zero
    if (all(w == 0)) break
    
    # Normalize
    w_norm <- sqrt(sum(w^2))
    if (w_norm == 0) break
    w <- w / w_norm
    
    # Convergence check
    if (sqrt(sum((w - w0)^2)) < tol) break
    
    w0 <- w
    iter <- iter + 1
  }
  })
  
  return(list(w = w, iter = iter, time = runningtime[[3]]))
}
