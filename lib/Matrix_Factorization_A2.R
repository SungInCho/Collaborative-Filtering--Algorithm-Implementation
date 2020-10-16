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
gradesc <- function(f = 10, 
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
  
  #Make matrix of u x i fill in 1 if user u did rate movie i otherwise put 0
  I_ui = matrix(0, U, I)
  for (s in train){
   I_ui[train[s,1],train[s,2]] = 1
  }
  
  train_RMSE <- c()
  test_RMSE <- c()
  
  
  # Math behind work 
  #P_u = P_u + lrate (1/2(2 I_ui(r_ui-T(q_i)*P_u)(P_u))+constant* 2P_u) # Implmentation  of A2
  
  
  #P_u = P_u + lrate ((r_ui-T(q_i)*P_u)*P_u)- lambda P_u) - Implementation of A1
  
  
  
  
  
  for(l in 1:max.iter){
    #random order of numbers from 1 to train rows without replacement, with number of trains output
    sample_idx <- sample(1:nrow(train), nrow(train))
    #loop through each training case and perform update
    

    for (movie in 1:I){
      
      i <- as.character(train[movie,2])
      
      for (user in 1:U){
        
        #Get the users, movie ids, and the ratings
        u <- as.character(train[user,1])
        
        #this is wrong, How do you find the right rating for user movie pair?
        r_ui <- train[(movie,user),3]
        
        # rating - Transpose column i of q maxtrix multipled by column u of p
        # rating(1x1) - the multiplication of a user and some movie (should be 1xf * fx1 matrix results in 1x1 matrix)
        # e_ui = 1x1
        e_ui <- r_ui - t(q[,i]) %*% p[,u]
        
        #error from rating times the user column - lambda  times movie column
        #gives gradient  
        # number* fx1 - lamba * fx1
        #grad_q = fx1
        # We are doing this grad_q and grad_p cause this is stochastic gradient decent
        
        #  t(q[,i]) %*% q[,i] is that supposed to be 2q[,i] or vice versa?
        # sig we can just test values 0.01 , .1, 1, or 10 (the sigs can be different per equation)
        
        sig=0.1
        grad_q <- 2*I_ui[user,movie] * e_ui %*% p[,u] + sig * 2*q[,i]
        
        if (all(abs(grad_q) > stopping.deriv, na.rm = T)){
          q[,i] <- q[,i] + 0.5*lrate * grad_q
        }
        grad_p <- 2*I_ui[user,movie] * e_ui %*% q[,i] + sig * 2*p[,u]
        
        if (all(abs(grad_p) > stopping.deriv, na.rm = T)){
          p[,u] <- p[,u] + 0.5*lrate * grad_p
        }
        
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
