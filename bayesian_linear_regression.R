library(dplyr)
library(ggplot2)
library(reshape2)

datapoints <- 1000
sigma.2 <- 1
sample.size <- 10000

df <- as.data.frame(cbind(rep(1, datapoints), matrix(rnorm(5*datapoints), ncol = 5)))
df <- mutate(df, y = V1 + V2 - V3 + V4 - V5 + V6 + rnorm(n = dim(df)[1], sd = sqrt(sigma.2)))

## Compare against lm implemetation
lm.fit <- lm(y ~ ., data = df)

beta.0.prior <- rnorm(n = sample.size)
beta.1.prior <- rnorm(n = sample.size)
beta.2.prior <- rnorm(n = sample.size)
beta.3.prior <- rnorm(n = sample.size)
beta.4.prior <- rnorm(n = sample.size)
beta.5.prior <- rnorm(n = sample.size)

prior <- as.matrix(cbind(beta.0.prior, beta.1.prior, beta.2.prior,
                         beta.3.prior, beta.4.prior, beta.5.prior))

myProb <- function(b, x){
    tmp <- b[7] - b[1]*x[1] - b[2]*x[2] - b[3]*x[3] - b[4]*x[4] - b[5]*x[5] - b[6]*x[6]
    tmp <- exp(-tmp / (2*pi*sigma.2)) / sqrt(2*pi*sigma.2)
    return(tmp)
}

myFunc <- function(a, data){
    tmp <- apply(data, MARGIN = 1, FUN = myProb, a)
    return(prod(tmp))
}

myFunc1 <- function(a, data){
    result = 1
    for(i in 1:dim(data)[1]){
        result <- result * myProb(data[i,], a)
    }
    return(result)
}

evidence <- apply(prior, MARGIN = 1, FUN = myFunc1, as.matrix(df))

posterior <- evidence * prior

myPlot <- function(df){
    tmp <- as.data.frame(df[, -c(1)])
    tmp <- melt(tmp)
    ggplot(data = tmp, aes(x=value)) + geom_histogram() + facet_wrap(~variable, scales="free")
}
