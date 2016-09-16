library(dplyr)
library(ggplot2)
library(reshape2)

datapoints <- 1000
sigma.var <- 1
sample.size <- 50000
prior.var <- 1

df <- as.data.frame(cbind(rep(1, datapoints), matrix(rnorm(5*datapoints), ncol = 5)))
df <- mutate(df, y = V1 + V2 - V3 + V4 - V5 + V6 + rnorm(n = dim(df)[1], sd = sqrt(sigma.var)))

## Compare against lm implemetation
lm.fit <- lm(y ~ . - 1, data = df)

myPlot <- function(df){
    tmp <- as.data.frame(df[, -c(1)])
    tmp <- melt(tmp)
    ggplot(data = tmp, aes(x=value)) + geom_density() + facet_wrap(~variable, scales="free")
}

generate.candidate <- function(b, std){
    return(rnorm(n = length(b), mean = b, sd = std))
}

acceptance.ratio <- function(cand.sample, init.sample, y, X, cand.var){
    cand.sample <- as.matrix(cand.sample)
    init.sample <- as.matrix(init.sample)
    log.alpha <- t(y - X %*% init.sample) %*% (y - X %*% init.sample) / (2 * sigma.var) +
        t(init.sample) %*% init.sample / (2 * prior.var) +
        t(cand.sample - init.sample) %*% (cand.sample - init.sample) / (2 * cand.var) -
        t(y - X %*% cand.sample) %*% (y - X %*% cand.sample) / (2 * sigma.var) -
        t(cand.sample) %*% cand.sample / (2 * prior.var) -
        t(init.sample - cand.sample) %*% (init.sample - cand.sample) / (2*cand.var)    
    return(exp(log.alpha))
}

main <- function(cand.std){
    posterior.sample <- matrix(rep(0, sample.size * 6), ncol = 6)
    y <- as.matrix(df$y)
    X <- as.matrix(df[, 1:6])

    for (i in 1:sample.size){
        ## Initialise sample
        if(i == 1){
            posterior.sample[1, ] <- rnorm(n = 6, sd = sqrt(prior.var))
            next
        }
        b <- posterior.sample[i-1,]
        b.prime <- generate.candidate(b, cand.std)
        alpha <- acceptance.ratio(b.prime, b, y, X, cand.std ** 2)
        if(alpha >= 1){
            posterior.sample[i, ] <- b.prime
        } else if(runif(n=1) <= alpha){
            posterior.sample[i, ] <- b.prime
        } else {
            posterior.sample[i, ] <- b
        }    
    }
    return(posterior.sample)
}
