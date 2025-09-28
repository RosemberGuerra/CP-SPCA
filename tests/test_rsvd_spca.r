## rSVD-sPCA function test

# install package if it does not exist yet
# install.packages('ltsspca') 

library(ltsspca)

# trying the function on the iris dataset
X <- iris[, 1:4]  

X_std = scale(X,center = T, scale = T)

result_raw = sPCA_rSVD(x = X, k = 2, method = "hard", center = FALSE, scale = FALSE,
          l.search = c(2,2), ls.min = NULL)
# l.search: the number of non-zero coefficients per component is
# ls.min: argument used only if the level of sparsity is unknown; it's used for model selection

result_raw$loadings
# > result_raw$loadings
# [,1]       [,2]
# [1,] -0.8257718  0.0000000
# [2,]  0.0000000 -0.9310918
# [3,] -0.5640043  0.0000000
# [4,]  0.0000000 -0.3647850

# resulting loadings are of length 1
t(result_raw$loadings ) %*% result_raw$loadings
# [,1] [,2]
# [1,]    1    0
# [2,]    0    1

# Altering the function such that the running time is included as a result
sPCA_rSVD_timed = function (x, k, method = "hard", center = FALSE, scale = FALSE, 
            l.search = NULL, ls.min = 1) {
  
    x <- as.matrix(x)
    n <- dim(x)[1]
    p <- dim(x)[2]
    v <- matrix(NA, p, k)
    Pb <- matrix(NA, p, k)
    xs <- scale(x, center = center, scale = scale)
    xe <- xs
    ls <- list()
    spca.it <- list()
    l.search0 <- l.search
    
    runningtime <- system.time(
      {
        
        for (iter in 1:k) {
          
          # if the number of nonzero weights is unknown
          # this scenario is irrelevant for our study
          if (is.null(l.search0)) {
            l.max <- p
            l.min <- 1
            l.s <- Inf
            while (l.s > ls.min) {
              l.search <- unique(round(seq(l.max, l.min, length.out = 10)))
              spca.it[[iter]] <- ltsspca:::sPCA.iterate(xe, method = method, 
                                              k = 1, l.search = l.search)
              ix.bic <- which.min(spca.it[[iter]]$bic)
              l.max <- l.search[max((ix.bic - 1), 1)]
              l.min <- l.search[min((ix.bic + 1), length(l.search))]
              l.s <- (l.max - l.min + 1)/10
            }
            l.search <- seq(l.max, l.min, by = -ls.min)
            spca.it[[iter]] <- ltsspca:::sPCA.iterate(xe, method = method, 
                                            k = 1, l.search = l.search)
          }
          
          # we specify the number of non-zero weights, so the following operation is executed
          else {
            l.search <- l.search0[[iter]]
            spca.it[[iter]] <- ltsspca:::sPCA.iterate(xe, method = method, 
                                            k = 1, l.search = l.search)
          }
          ls[[iter]] <- l.search
          v[, iter] <- spca.it[[iter]]$v
          xe <- ltsspca:::Deflate.PCA(xe, v[, iter])
        }
        scores <- xs %*% v
        eigenvalues <- apply(scores, 2, var)
      }
    )
    
    return(list(loadings = v, scores = scores, eigenvalues = eigenvalues, 
                ls = ls, spca.it = spca.it, runningtime = runningtime))
}

result_timed = sPCA_rSVD_timed(x = X, k = 2, method = "hard", center = FALSE, scale = FALSE,
                            l.search = c(2,2), ls.min = NULL)

result_timed$loadings

result_timed$runningtime
