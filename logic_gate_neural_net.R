## Toy project creating simple neural networks to create logic gates

and.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 0, 0, 1))
or.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 1, 1, 1))
xor.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 1, 1, 0))

sigmoid.func <- function(x){
    return(1 / (1 + exp(-x)))
}

fwd.prop <- function(x, weights){
    n.1 <- sigmoid.func(x %*% weights[[1]])
    n.2 <- sigmoid.func(n.1 %*% weights[[2]])
    return(list(n.1, n.2))
}

grad.back.prop <- function(output, input, layers, weights){
    delta.2 <- (layers[[2]] - output) * layers[[2]] * (1- layers[[2]])
    grad.2 <- matrix(outer(delta.2, layers[[1]])[,,1,], ncol = length(delta.2), byrow = TRUE)
        
    delta.1 <- t(weights[[2]] %*% delta.2) * layers[[1]] * (1 - layers[[1]])
    grad.1 <- matrix(outer(delta.1, input)[,,1,], ncol = length(delta.1), byrow = TRUE)

    return(list(grad.1, grad.2))
}

update.weights <- function(weights, grads, step.size){
    weights[[1]] <- weights[[1]] - step.size * grads[[1]]
    weights[[2]] <- weights[[2]] - step.size * grads[[2]]
    return(weights)
}

train.error <- function(y, X, weights){
    layers <- fwd.prop(X, weights)
    err <- sum((y - layers[[2]]) ** 2) / length(y)
    return(err)
}

step.size <- 0.05

## Two layer neural net
##
## First has 3 neurons layer, takes 2 inputs thus has 6 weighting parameters
## Output layer has single neuron, thus has 3 weight parameters


main <- function(data, step.size, iterations, reports){

    X <- as.matrix(data[, 1:2])
    y <- as.matrix(data$y)

    ## Initialise weights
    W <- matrix(rnorm(n = 6, sd = 0.01), ncol = 3)
    U <- matrix(rnorm(n = 3, sd = 0.01), ncol = 1)
    initial.weights <- list(W, U)
    weights <- initial.weights
    print(sprintf("Initial training error: %s",train.error(y, X, weights)))
    
    for(s in 1:iterations){
        i <- (s %% 4) + 1
                
        ## Input matrix 1x2, 2 inputs)
        input <- X[i, , drop = FALSE] # REMEMBER TO CHANGE BACK
        output <- y[i, , drop = FALSE]

        layers <- fwd.prop(input, weights)
        
        grads <- grad.back.prop(output, input, layers, weights)
        
        weights <- update.weights(weights, grads, step.size)

        if(s %% (iterations %/% reports) == 0){
            print(sprintf("Iteration: %s", s))
            print(sprintf("Training error: %s",train.error(y, X, weights)))
        }
    }

    print(sprintf("Final training error: %s",train.error(y, X, weights)))
    return(list(weights, initial.weights))
}

################## Accuracy below ################
## xor gate, step-size = 0.5, 1 million iterations
## accuracy 0.0001
## output
##           [,1]
##[1,] 0.01323084
##[2,] 0.98958391
##[3,] 0.98958456
##[4,] 0.01448111

##################################################
## and gate, step-size = 0.5, 1 million iterations
## accuracy 0.063
## output
##             [,1]
##[1,] 0.0000417549
##[2,] 0.0026254057
##[3,] 0.0028121358
##[4,] 0.4999494329

##################################################
## or gate, step-size = 0.5, 1 million iterations
## accuracy 0.000009
## output
##            [,1]
##[1,] 0.004987676
##[2,] 0.997537258
##[3,] 0.997535449
##[4,] 0.998335838

##################################################
## Using neuralnet package to compare performance essentially to check implementation

## This produces correct results, but includes constant "bias" neurons which makes the
## comparison difficult
library(neuralnet)
nn <- neuralnet(y ~ x1 + x2 - 1, data = and.gate, hidden = c(3), algorithm = "backprop",
                learningrate = 0.05, err.fct = "sse", startweights = rnorm(n = 13, sd = 0.01))
compute(nn, and.gate[, 1:2])
