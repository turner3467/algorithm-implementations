## Toy project creating simple neural networks to create logic gates

and.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 0, 0, 1))
or.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 1, 1, 1))
xor.gate <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1), y = c(0, 1, 1, 0))

X <- as.matrix(and.gate[, 1:2])
y <- as.matrix(and.gate$y)

sigmoid.func <- function(x){
    return(1 / (1 + exp(-x)))
}

fwd.prop <- function(X, W, U){
    n.1 <- sigmoid.func(W %*% t(X))
    n.2 <- sigmoid.func(U %*% n.1)
    return(n.2)
}

train.error <- function(y, n.2){
    err <- sum((y - t(n.2)) ** 2) / length(y)
    return(err)
}

step.size <- 0.05

## Two layer neural net
##
## First has 3 neurons layer, takes 2 inputs thus has 6 weighting parameters
## Output layer has single neuron, thus has 3 weight parameters


main <- function(step.size, iteration){
    
    ## Initialise weights
    ## W is matrix of weights for first layer, 3x2
    W <- matrix(rnorm(n = 6, sd = 0.01), ncol = 2)

    ## U is matrix of weights for second layer, 1x3
    U <- matrix(rnorm(n = 3, sd = 0.01), ncol = 3)

    for(s in 1:1000){
        j <- (s %% 6) + 1
        if(j >= 4){
            i <- 4
        } else {
            i <- j
        }
        
        ## Input matrix 2x1, 2 inputs)
        input <- as.matrix(X[i, ])
        output <- y[i, ]

        ## N.1 first layer of neurons, 3x1
        N.1 <- sigmoid.func(W %*% input)

        ## N.2 output layer, single neuron 1x1
        N.2 <- sigmoid.func(U %*% N.1)

        ## Gradient for U weights, 3x1
        delta.U <- as.vector((N.2 - output) * N.2 * (1 - N.2))
        grad.U <- outer(delta.U, N.1)[,, 1]

        ## Gradient for W weights
        delta.W <- as.vector(sum(delta.U * U) * N.1 * (1 - N.1))
        grad.W <- outer(delta.W, input)[,, 1]
        
        W <- W - step.size * grad.W
        U <- U - step.size * grad.U
    }

    print("Training error is: ")
    print(train.error(y, fwd.prop(X, W, U)))
    return(fwd.prop(X, W, U))
}

## Using neuralnet package to compare performance essentially to check implementation

## This produces correct results, but includes constant "bias" neurons which makes the
## comparison difficult
library(neuralnet)
nn <- neuralnet(y ~ x1 + x2 - 1, data = and.gate, hidden = c(3), algorithm = "backprop",
                learningrate = 0.05, err.fct = "sse", startweights = rnorm(n = 13, sd = 0.01))
compute(nn, and.gate[, 1:2])
