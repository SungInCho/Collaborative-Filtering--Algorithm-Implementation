#Define a function to calculate RMSE

#Summation from 1 to n of 
#(sqrt((observe_i-estimated_i)^2/n)

RMSE <- function(rating, est_rating){
  #(square error = (observe - estimated)^2)
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[2]), as.character(obs[1])])^2
    return(sqr_error)
  }
  # to all the ratings by row, run the function sqr_err
  # take the mean then sqrt the result
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

#Stochastic Gradient Descent
# a function returns a list containing factorized matrices p and q, training and testing RMSEs.
gradesc_A2 <- function(f = 10, 
                    lambda = 0.3,lrate = 0.01, max.iter, stopping.deriv = 0.01,
                    data, train, test){
  set.seed(0)
  
  #random assign value to matrix p and q
  #p is a matrix that generates f*U (f*Number of Users) numbers that are between -1 and 1 with U columns
  #p has U columns and f rows (column names are numbers)
  p <- matrix(runif(f*U, -1, 1), ncol = U) 
  colnames(p) <- as.character(1:U)
  
  #q is a matrix that generates f*I (f*Number of Movies) numbers that are between -1 and 1 with I columns
  #q has I columns and f rows (column names are movieids?)
  q <- matrix(runif(f*I, -1, 1), ncol = I)
  colnames(q) <- levels(as.factor(data$movieId))
  
  train_RMSE <- c()
  test_RMSE <- c()
  
  
  for(l in 1:max.iter){
    sample_idx <- sample(1:nrow(train), nrow(train))
    #loop through each training case and perform update
    
    #for each element in our train sample(columns being userId,movieId,rating,timestamp)
    for (s in sample_idx){
      
      #Get the users, movie ids, and the ratings
      u <- as.character(train[s,1])
      
      i <- as.character(train[s,2])
      
      r_ui <- train[s,3]
      
      # rating - Transpose column i of q maxtrix multipled by column u of p
      # rating(1x1) - the multiplication of a user and some movie (should be 1xf * fx1 matrix results in 1x1 matrix)
      # e_ui = 1x1
      e_ui <- r_ui - t(q[,i]) %*% p[,u]
      
      #error from rating times the user column - lambda  times movie column
      #gives gradient  
      # number* fx1 - lamba * fx1
      #grad_q = fx1
      # We are doing this grad_q and grad_p cause this is stochastic gradient decent
      grad_q <- e_ui %*% p[,u] - lambda * q[,i]
      
      if (all(abs(grad_q) > stopping.deriv, na.rm = T)){
        q[,i] <- q[,i] + lrate * grad_q
      }
      grad_p <- e_ui %*% q[,i] - lambda * p[,u]
      
      if (all(abs(grad_p) > stopping.deriv, na.rm = T)){
        p[,u] <- p[,u] + lrate * grad_p
      }
    }
    #print the values of training and testing RMSE
    if (l %% 10 == 0){
      cat("epoch:", l, "\t")
      est_rating <- t(q) %*% p
      rownames(est_rating) <- levels(as.factor(data$movieId))
      
      train_RMSE_cur <- RMSE(train, est_rating)
      cat("training RMSE:", train_RMSE_cur, "\t")
      train_RMSE <- c(train_RMSE, train_RMSE_cur)
      
      test_RMSE_cur <- RMSE(test, est_rating)
      cat("test RMSE:",test_RMSE_cur, "\n")
      test_RMSE <- c(test_RMSE, test_RMSE_cur)
    } 
  }
  
  return(list(p = p, q = q, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
}