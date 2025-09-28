########################################################
# Numerical Experiment: Trade-Off Curve of Variance vs. Sparsity #
# Author: RI Guerra Urzola                             #
# Date: 28-09-2025                                     #
########################################################

# Set the working directory to the location of this file
# (i am unable to change the working directory; I'll work with the working directory = "CP-SPCA")
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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
load('./data/synthetic/setup.RData')

method <- c('alt','cp', 'rsvd') # Types of penalty functions
k <- 1                             # Number of components

# Create a grid of parameters for the simulations
param_grid <- expand.grid(n = n, p = p, s = c(1:S), method = method)
# param_grid EXCLUDES the alpha values because we will define the alpha parameter different across 3 methods

total_iterations <- nrow(param_grid)  # Total number of iterations

# alpha for alternating
n_alpha_alt <- 10                      # Number of regularization parameters for alternating
alpha_alt <- seq(from = 0, to = 1, by = 1 / n_alpha_alt) # Regularization parameters for alternating

# alpha for CP-PCA
n_alpha_cp <- 10                      # Number of regularization parameters for CP-PCA
alpha_cp <- seq(from = 0, to = 0.1, by = 1 / n_alpha_cp) # Regularization parameters for CP-PCA

# cardinality for sPCA-rSVD
n_card_rsvd <- 10                     # Number of different cardinality values for sPCA-rSVD


#### Results Initialization ####

# Create an empty data frame to store the results
results <- data.frame()
# Source the script that contains the functions (to ensure availability in each worker)
# Note here that I'm using "./code/" instead of "../code/", because i could not change the working directory

source('./code/alternating_spca.R')
source('./code/cardinality_penalty_spca.R')
source('./code/spca_rsvd.r')
source('./code/metrics.R')

#### Simulation Loop with Parallelization ####

# Perform parallel computation for each parameter set in the grid
for(i in 1:total_iterations){

  # Extract parameter values for the current iteration
  n <- param_grid$n[i]
  p <- param_grid$p[i]
  s <- param_grid$s[i]
  
  method = param_grid$method[i]
  
  # Load the corresponding dataset
  
  # Note here also the change in the path!
  X <- read.table(paste0('./data/synthetic/simdata_', n, '_', p, '_', s, '.txt')) %>% as.matrix()
  if (method == "alt"){
    # Alternating spca #
    spca <- alt_spca(X, alpha = param_grid$alpha[i], penalty = "l0")  
  }
  if (method == "cp"){
    spca <- cp_spca(X, alpha = param_grid$alpha[i])
  }
  if (method == 'rsvd'){
    # argument explained:
    # x = data
    # k = number of component
    # method = thresholding type
    # center, scale = mean-centering, scaling
    # l.search = vector; desired cardinality level per component
    # ls.min = redundant for our use
    
    # So here, I'd suggest tweaking the simulation such that you can control for the
    # n_card_rsvd below here!
    spca <- sPCA_rSVD_timed(x = X, k = 1, method = 'hard', 
                            center = F, scale = F, 
                            l.search = c(n_card_rsvd), ls.min = NULL)
  }
  
  
  
  # Calculate performance metrics
  pev <- variance(X, spca$w)      # Proportion of variance explained
  pev_adj <- adj_variance(X, spca$w) # Adjusted proportion of variance explained
  card <- cardinality(spca$w)    # Sparsity (number of selected features)
  
  # Return results for the current iteration
  results <- rbind(results, data.frame(param_grid[i, ], pev, card, pev_adj, spca$time))
}

