cv.function2 <- function(data,dat_train, K,epochs,lrate, f, lambda_p, lambda_q){
  ### Input:
  ### - train data frame
  ### - K: a number stands for K-fold CV
  ### - tuning parameters 
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  train_rmse <- matrix(NA, ncol = epochs/10,nrow = K)
  test_rmse <- matrix(NA, ncol = epochs/10, nrow = K)
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- gradesc2(f = f, lambda_p =lambda_p,lambda_q=lambda_q,
                      lrate = lrate, epochs = epochs, data=data,
                      train = train.data, test = test.data)
    
    train_rmse[i,] <-  result$train_RMSE
    test_rmse[i,] <-   result$test_RMSE
    
  }		
  return(list(mean_train_rmse = apply(train_rmse, 2, mean), mean_test_rmse = apply(test_rmse, 2, mean),
              sd_train_rmse = apply(train_rmse, 2, sd), sd_test_rmse = apply(test_rmse, 2, sd)))
}
