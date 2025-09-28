## rSVD-sPCA function test

# install package if it does not exist yet
install.packages('ltsspca') 

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
ss
# resulting loadings are of length 1
t(result_raw$loadings ) %*% result_raw$loadings
# [,1] [,2]
# [1,]    1    0
# [2,]    0    1
