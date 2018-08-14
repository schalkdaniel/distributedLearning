
## Create Test Datasets

``` r
library(nycflights13)

df.train = na.omit(as.data.frame(flights))

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

myformula = formula(arr_delay ~ month + air_time)

mod.lm = lm(myformula, data = df.train)
summary(mod.lm)
## 
## Call:
## lm(formula = myformula, data = df.train)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
##  -90.5  -23.9  -11.7    7.2 1272.0 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)
## (Intercept) 10.954894   0.220325   49.72   <2e-16
## month       -0.234965   0.024033   -9.78   <2e-16
## air_time    -0.016799   0.000876  -19.19   <2e-16
## 
## Residual standard error: 44.5 on 294609 degrees of freedom
## Multiple R-squared:  0.00159,    Adjusted R-squared:  0.00158 
## F-statistic:  234 on 2 and 294609 DF,  p-value: <2e-16

mybeta = lmGradientDescent(myformula, data = df.train, iters = 300000L, learning_rate = 0.001,
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

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!

## Warning in lmGradientDescent_internal(X, data[[response]], iters = iters, :
## Be careful, mse is not improving! Reducing the learning rate by 20 percent!
```

|             |       lm | grad.descent |     diff |
| ----------- | -------: | -----------: | -------: |
| (Intercept) |  10.9549 |       9.9413 |   1.0136 |
| month       | \-0.2350 |     \-0.1561 | \-0.0789 |
| air\_time   | \-0.0168 |     \-0.0144 | \-0.0024 |

## Distributed Linear Model

``` r
files = paste0("data/", c("flights1.csv", "flights2.csv", "flights3.csv"))
```
