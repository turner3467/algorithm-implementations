library(dplyr)

df <- as.data.frame(matrix(rnorm(5000), ncol = 5))
df <- mutate(df, y = 5 + V1 + 2*V2 - V3 + 3*V4 + V5 + rnorm(n = dim(df)[1]))

## Compare against lm implemetation
lm.fit <- lm(y ~ ., data = df)

df <- mutate(df, V0 = 1)

myLR <- function(df){
    y.mat <- as.matrix(df$y)
    x.mat <- as.matrix(subset(df, select = -c(y)))
    beta.hat <- solve(t(x.mat) %*% x.mat) %*% t(x.mat) %*% y.mat
    y.hat <- x.mat %*% beta.hat
    sigma.2.hat <- sum((y.mat - y.hat) ^ 2) / (length(y.mat) - dim(x.mat)[2] - 1)
    se.beta.hat <- diag(sqrt(solve(t(x.mat) %*% x.mat) * sigma.2.hat))
    rse <- sqrt(t(y.mat - y.hat) %*% (y.mat - y.hat) / (length(y.mat) - dim(x.mat)[2] - 1))
    rss <- sum((y.mat - y.hat)^2)
    tss <- sum((y.mat - mean(y.mat))^2)
    r.2 <- 1 - (rss / tss)
    print("Coefficients")
    print(beta.hat)
    print("Standard error of Coefficients")
    print(se.beta.hat)
    print("Residual standard error")
    print(rse)
    print("R-squared")
    print(r.2)
    return(0)
}
