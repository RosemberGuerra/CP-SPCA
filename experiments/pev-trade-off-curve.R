########################################################
# Numerical Experiment: Trade-Off Curve of Variance vs. Sparsity #
# Author: RI Guerra Urzola                             #
# Date: 28-09-2025                                     #
########################################################

# Set the working directory to the location of this file
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#### Load required libraries ####
library(foreach)      # Parallel processing support
library(doParallel)   # Parallel backend for foreach
library(dplyr)        # Data manipulation
library(ggplot2)      # Data visualization
library(progressr)    # Progress tracking

# Clear the workspace to avoid conflicts with previous variables or functions
rm(list = ls())


# Set seed for reproducibility
set.seed(123)


#### Parallel Processing Setup ####

num_cores <- detectCores() - 2  # Use all but two cores for parallelization
cl <- makeCluster(num_cores)
registerDoParallel(cl)


#### Parameter Initialization ####
load('../data/synthetic/setup.RData')

methd <- c('alt','cp') # Types of penalty functions
k <- 1                             # Number of components
n_alpha <- 10                      # Number of regularization parameters
alpha <- seq(from = 0, to = 1, by = 1 / n_alpha) # Regularization parameters

# Create a grid of parameters for the simulations
param_grid <- expand.grid(n = n, p = p, s = c(1:S), method = methd, alpha = alpha)
total_iterations <- nrow(param_grid)  # Total number of iterations


#### Results Initialization ####

# Create an empty data frame to store the results
results <- data.frame()
# Source the script that contains the functions (to ensure availability in each worker)
source('../code/alternating_spca.R')
source('../code/cardinality_penalty_spca.R')
# rSVD-sPCA 
source('../code/metrics.R')

#### Simulation Loop with Parallelization ####

# Perform parallel computation for each parameter set in the grid
for(i in 1:total_iterations){

  # Extract parameter values for the current iteration
  n <- param_grid$n[i]
  p <- param_grid$p[i]
  s <- param_grid$s[i]
  
  method = param_grid$method[i]
  
  # Load the corresponding dataset
  X <- read.table(paste0('../data/synthetic/simdata_', n, '_', p, '_', s, '.txt')) %>% as.matrix()
  if (method == "alt"){
    # Alternating spca #
    spca <- alt_spca(X, alpha = param_grid$alpha[i], penalty = "l0")  
  }
  if (method == "cp"){
    spca <- cp_spca(X, alpha = param_grid$alpha[i])
  }
  
  
  # Calculate performance metrics
  pev <- variance(X, spca$w)      # Proportion of variance explained
  pev_adj <- adj_variance(X, spca$w) # Adjusted proportion of variance explained
  card <- cardinality(spca$w)    # Sparsity (number of selected features)
  
  # Return results for the current iteration
  results[i]= data.frame(param_grid[i, ], pev, card, pev_adj, spca$time)
}

