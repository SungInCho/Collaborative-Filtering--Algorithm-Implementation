unroll_Vecs <- function (params, Y, R, num_users, num_movies, num_features) {
  # Unrolls vector into X and Theta
  # Also calculates difference between preduction and actual 
  
  endIdx <- num_movies * num_features
  
  X     <- matrix(params[1:endIdx], nrow = num_movies, ncol = num_features)
  Theta <- matrix(params[(endIdx + 1): (endIdx + (num_users * num_features))], 
                  nrow = num_users, ncol = num_features)
  
  Y_dash     <-   (((X %*% t(Theta)) - Y) * R) # Prediction error
  
  return(list(X = X, Theta = Theta, Y_dash = Y_dash))
}

J_cost <-  function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  # Calculates the cost
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  J <-  .5 * sum(   Y_dash ^2)  + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)
  
  return (J)
}

grr <- function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  # Calculates the gradient step
  # Here lambda is the regularization parameter
  # Alpha is the step size
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  X_grad     <- ((   Y_dash  %*% Theta) + lambda * X     )
  Theta_grad <- (( t(Y_dash) %*% X)     + lambda * Theta )
  
  grad = c(X_grad, Theta_grad)
  return(grad)
}

# Now that everything is set up, call optim
print(
  res <- optim(par = c(runif(num_users * num_features), runif(num_movies * num_features)), # Random starting parameters
               fn = J_cost, gr = grr, 
               Y=Y, R=R, 
               num_users=num_users, num_movies=num_movies,num_features=num_features, 
               lambda=lambda, alpha = alpha, 
               method = "L-BFGS-B", control=list(maxit=maxit, trace=1))
)