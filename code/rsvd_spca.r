## rSVD-sPCA function implementation

# install package if it does not exist yet
if (!requireNamespace('ltsspca', quietly = TRUE)) {
  install.packages(ltsspca)
}

library(ltsspca)

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

