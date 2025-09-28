############################################
# Data Generation for Simulation Exercise  #
# Author: RI Guerra Urzola                 #
# Date: 28-09-2024                         #
############################################

# Clear the workspace to avoid conflicts with previous variables or functions
rm(list = ls())

# Load external functions for data generation
generate_ordered_eigen_data <- function(n, p, mu = 0, sigma = 1, eigenvalues = NULL) {
  # n: number of samples
  # p: number of features
  # mu: mean of the normal distribution
  # sigma: standard deviation of the normal distribution
  # eigenvalues: optional vector of decreasing eigenvalues
  
  if (is.null(eigenvalues)) {
    # Generate a decreasing sequence of positive eigenvalues
    if (p > n){
      eigenvalues <- c(sort(runif(n, .5, 10), decreasing = TRUE),
                       rep(0,p-n))
    }else {
      eigenvalues <- sort(runif(p, .5, 10), decreasing = TRUE)  
    }
    
  }
  
  # Generate a random orthonormal matrix (eigenvectors)
  Q <- qr.Q(qr(matrix(rnorm(p^2), p, p)))
  
  # Construct the covariance matrix with controlled eigenvalues
  Sigma <- Q %*% diag(eigenvalues) %*% t(Q)
  
  # Generate multivariate normal data
  library(MASS)
  X <- mvrnorm(n, rep(mu, p), Sigma)
  
  return(X)
}

# Set the seed for reproducibility of random number generation
set.seed(123)

############################################
# Simulation Parameters                    #
############################################
# S: Number of simulations to run
# n: Number of observations in each dataset
# p: Number of variables to test (small, medium, large settings)
S <- 10
n <- 100
p <- c(20)

# Create a grid of parameter combinations for simulations
param_grid <- expand.grid(n = n, p = p, s = 1:S)

############################################
# Data Distribution Parameters             #
############################################
# mu: Mean of the normal distribution
# sigma: Standard deviation of the normal distribution
mu <- 0
sigma <- 1

############################################
# Progress Bar Setup                       #
############################################
# Initialize a progress bar to track simulation progress
pb <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)

############################################
# Simulation Loop                          #
############################################
# For each combination of parameters, generate and save a dataset
for (i in 1:nrow(param_grid)) {
  # Update the progress bar
  setTxtProgressBar(pb, i)
  
  # Extract current parameter settings
  n <- param_grid$n[i]
  p <- param_grid$p[i]
  s <- param_grid$s[i]
  
  # Generate a data matrix with normal distribution
  X <- generate_ordered_eigen_data(n, p, mu, sigma)
  
  # Scale the data matrix to have zero mean and unit variance
  X <- scale(X)
  
  # Save the generated data matrix to a file
  write.table(X, 
              file = paste0('../data/synthetic/simdata_', n, '_', p, '_', s, '.txt'), 
              row.names = FALSE, 
              col.names = FALSE)
}

# Close the progress bar after all simulations are completed
close(pb)
