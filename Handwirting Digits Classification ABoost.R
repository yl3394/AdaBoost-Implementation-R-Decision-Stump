#=========================================================================
#STAT W4400 
#Homework 03 
# yl3394, Yanjin Li 
#Problem 1 AdaBoost 
#=========================================================================
# In this problem I will implement AdaBoost algorithm in R. The algorithm 
# requires two auxiliary functions, to train and to evaluate the weak leaner.
# And, then we will have the third function for implementing the resulting 
# boosting classifier. Here, we will use the decision stumps as our weak 
# learners. 
#=========================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0.  Set-ups
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
install.packages("ada")
library("rpart")
library("ada")
uspsdata <- read.table(file.choose(), header = F)
uspscl <- read.table(file.choose(), header = F)
uspsdata <- as.matrix(uspsdata)
uspscl <- as.matrix(uspscl)
n <- nrow(uspsdata)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1.  Decision Stump: Weak Learner Training Routine 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# w,y are vectors containing the weights and class labels, the output pars
# is a list which contains the parameters specifying the result classifiers.
# In this part, I will train the decision stump by between-axis and within-
# axis comparisons. 

train <- function(X, w, y){
  n <- nrow(X)
  p <- ncol(X)
  elist <- matrix(nrow=p)
  mlist <- matrix(nrow=p)
  tlist <- matrix(nrow=p)
  for(j in 1:p){
    # Sort data points along dimension 
    indx <- order(X[,j])
    x_j <- X[indx, j]
    # Using a cummulative sum, count the weight when progressively shifting 
    # the threshold to the right 
    w_cum <- cumsum(w[indx] * y[indx]) 
    
    # Handle multiple occurences of same x_j values 
    # threshold point must not lie between elements of the same value 
    w_cum[duplicated((x_j)==1)] <- NA
    
    # Find the optimal threshold and classify accordingly 
    m <- max(abs(w_cum), na.rm = TRUE)  #remove RM 
    print(m)
    maxIndx <- min(which(abs(w_cum)==m))
    mlist[j] <- (w_cum[maxIndx]<0)*2 - 1  
    tlist[j] <- x_j[maxIndx]
    c <- ((x_j > tlist[j])*2 -1) * mlist[j]  
    #here (...>...) will give us 1 if is true,otherwise 0 
    elist[j] <- w %*% (c!=y) 
  }
  m <- min(elist)
  j_star <- min(which(elist==m))
  
  pars <- list(j=j_star, theta = tlist[j_star], mode = mlist[j_star])
  return(pars)
}

classify <- function(X, pars){
  j <- pars$j
  t <- pars$theta
  m <- pars$mode
  x <- X[,j]
  pred <- m * (x-t)
  pred[pred < 0] <- -1
  pred[pred >= 0] <- 1
  return(pred)
}


agg_class <- function(X, alpha, allPars){
  pred_agg <- vector(length = nrow(X))
  M <- length(alpha)
  for(m in 1:M){
    pred_agg <- pred_agg + (alpha[m] * classify(X, allPars[m])) 
  }
  pred_agg[pred_agg >= 0] <- 1
  pred_agg[pred_agg < 0] <- -1
  return(pred_agg)
}

adaBoost <- function(X, y, B){
  n <- nrow(X)
  w <- rep(1/n, times = n)
  alpha <- rep(0,times=B)
  allPars <- rep(list(list()),B)
  for(b in 1:B){
    # Step 1: train base classifier 
    allPars[[b]] <- train(X, w, y)
    # Step 2: compute error
    misClass <- (y!= classify(X, allPars[[b]]))
    e <- (w %*% misClass/sum(w))
    # Step 3: compute voting weight 
    alpha[b] <- log((1-e)/e)
    # Step 4: recompute weights 
    w <- w * exp(alpha[b] * misClass)
  }
  return(list(allPars=allPars, alpha=alpha))
}

## File to run AdaBoost

B_max <- 60
nCV <- 5 
n <- nrow(uspsdata)

testErrorRate <- matrix(0, nrow = B_max, ncol = nCV)
trainErrorRate <- matrix(0, nrow = B_max, ncol = nCV)
for(i in 1:nCV){
  # Randomly split data in training and test half 
  n <- nrow(uspsdata)
  p <- sample.int(n)
  trainIndx <- p[1:round(n/2)]
  testIndx <- p[-(1:round(n/2))]
  
  ada <-adaBoost(uspsdata[trainIndx,], uspscl[trainIndx], B_max)
  allPars <- ada$allPars 
  alpha <- ada$alpha 
  
  # Determine error rate, depending on the number of base classifier 
  for(B in 1:B_max){
    c_hat_test <- agg_class(uspsdata[testIndx, ], alpha[1:B], allPars[1:B])
    testErrorRate[B,i] <- mean(uspscl[testIndx]!= c_hat_test)
    c_hat_train <- agg_class(uspsdata[trainIndx, ], alpha[1:B], allPars[1:B])
    trainErrorRate[B,i] <- mean(uspscl[trainIndx]!=c_hat_train)
  }
}

# Plot 
matplot(trainErrorRate ,type="l",lty=1:nCV, ylim = c(0,0.5))
matplot(testErrorRate ,type="l",lty=1:nCV, ylim = c(0,0.05))
