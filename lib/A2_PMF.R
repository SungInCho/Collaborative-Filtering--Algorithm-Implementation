RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (as.numeric(obs[3]) - est_rating[as.character(obs[2]), as.character(obs[1])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

gradesc2 <- function(f = 10, 
                    lrate = 0.01, lambda_p, lambda_q, epochs, print_every, data=data,
                    train, test){
  
  U <- length(unique(data$userId))
  I <- length(unique(data$movieId))
  
  train_RMSE <- c()
  test_RMSE <- c()
  epoch_dur <- c()
  
  set.seed(0)
  #random assign value to matrix p and q
  p <- matrix(runif(f*U, -1, 1), ncol = U) 
  colnames(p) <- as.character(1:U)
  
  q <- matrix(runif(f*I, -1, 1), ncol = I)
  colnames(q) <- levels(as.factor(data$movieId))
  t0 = Sys.time()
  
  for(l in 1:epochs){
      
    if (l%%print_every == 1){
      t1 = Sys.time()
    }
    
    grad_p <- matrix(0, ncol=U, nrow=f)
    colnames(grad_p) <- as.character(1:U)
    grad_q <- matrix(0, ncol=I, nrow=f)
    colnames(grad_q) <- levels(as.factor(data$movieId))
    
    for (i in 1:nrow(train)){
      user <- as.character(train[i,1])
      movie <- as.character(train[i,2])
      err <- as.numeric(train[i,3])- t(p[,user]) %*% q[,movie]
      error <- rep(err, f)
      grad_p[,user] <- grad_p[,user]+error*q[,movie]
      grad_q[,movie] <- grad_q[,movie]+error*p[,user]
      
    }
    
    for(i in 1:U){
      grad_p[,as.character(i)] <- grad_p[,as.character(i)]-lambda_p*p[,as.character(i)]
    }
    for(i in levels(as.factor(data$movieId))){
      grad_q[,i] <- grad_q[,i]-lambda_q*q[,i]
    }
    
    p <- p + lrate*grad_p
    q <- q + lrate*grad_q
    
  # aotomatically change learning rate to avoid diverge

    #print the values of training and testing RMSE every 10 epochs
    if (l %% print_every == 0){
    est <- t(q) %*% p
    rownames(est) <- levels(as.factor(data$movieId))
    train_RMSE_cur <- RMSE(train,est)
    test_RMSE_cur <- RMSE(test, est)
    t_cur <- difftime(Sys.time(), t1, units='mins')
    
    train_RMSE <- c(train_RMSE, train_RMSE_cur)
    test_RMSE <- c(test_RMSE, test_RMSE_cur)
    epoch_dur <- c(epoch_dur, t_cur)
      cat("epochs:", l, "\t")
      cat("training RMSE:", train_RMSE_cur, "\t")
      cat("test RMSE:",test_RMSE_cur, "\n")
      cat("epoch duration:", t_cur,"\t")
    }
  }
  t_total = difftime(Sys.time(), t0, units='mins')
  print(paste("Training is completed. This process took" ,t_total, "minutes to complete") )
  return(list(p = p, q = q, train_RMSE=train_RMSE, test_RMSE=test_RMSE))
}
