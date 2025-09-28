## Metrics ##


## Variance calculation
variance <- function(X, w) {
  w_pca <- prcomp(X)
  (t(w) %*% t(X) %*% X %*% w) / w_pca$sdev[1]^2
}

## Adjusted variance calculation
adj_variance <- function(X, w) {
  indexw <- which(w != 0)
  X_adj <- X[, indexw]
  w_adj <- prcomp(X_adj)$rotation[, 1]
  w_pca <- prcomp(X)
  (t(w_adj) %*% t(X_adj) %*% X_adj %*% w_adj) / w_pca$sdev[1]^2
}

## Cardinality (number of non-zero elements)
cardinality <- function(w) {
  sum(w != 0)
}
