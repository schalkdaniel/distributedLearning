
## Create Test Datasets

``` r
library(nycflights13)

df.train = as.data.frame(flights)

set.seed(314159)
idx.test = sample(x = seq_len(nrow(df.train)), size = 0.1 * nrow(df.train))

df.test  = df.train[idx.test, ]
df.train = df.train[-idx.test, ]

splits = 3L
breaks = seq(1, nrow(df.train), length.out = splits + 1)

# Save single train sets:
for (i in seq_len(splits)) {
    temp = df.train[breaks[i]:breaks[i + 1], ]
    write.csv(x = temp, file = paste0("data/flights", i, ".csv"), row.names = FALSE)
}

# Save test set:
write.csv(x = df.test, file = "data/flights_test.csv")
```

## Run Gradient Descent

``` r
Rcpp::sourceCpp("src/gradient_descent.cpp")
source("R/grad_descent_lm.R")

myformula = formula(Petal.Length ~ Sepal.Length + Petal.Width)

(mod.lm = lm(myformula, data = iris))
## 
## Call:
## lm(formula = myformula, data = iris)
## 
## Coefficients:
##  (Intercept)  Sepal.Length   Petal.Width  
##       -1.507         0.542         1.748

mybeta = lmGradientDescent(myformula, data = iris, iters = 10000L, learning_rate = 0.1,
    mse_eps = 1e-9, trace = FALSE)
## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!
```

|              |       lm | grad.descent |     diff |
| ------------ | -------: | -----------: | -------: |
| (Intercept)  | \-1.5071 |     \-1.4931 | \-0.0140 |
| Sepal.Length |   0.5423 |       0.5394 |   0.0029 |
| Petal.Width  |   1.7481 |       1.7504 | \-0.0023 |

## Distributed Linear Model

``` r
files = paste0("data/", c("flights1.csv", "flights2.csv", "flights3.csv"))
```
