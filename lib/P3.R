library(pracma) # library to use Norm

# Gaussian kernel (t_x_i: transpose of x_i)
K <- function(t_x_i, X){
  return(exp(2*(t_x_i%*%t(X))-1))
}

# Making X, predictor matrix
X_mat <- function(q){
  V <- t(q)
  X <- apply(V, 2, function(x) x/Norm(x)) # make X using v_j (mxk)
  return(X)
}

# SVD with kernel ridge regression 
svd_krr <- function(n, lamb=0.5, X, pred_rate){
  I <- diag(n)
  return(K(X,X) %*% solve((K(X,X)+lamb*I)) %*% pred_rating)
}
