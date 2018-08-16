
## Create Test Datasets

``` r
set.seed(314159)

df.train = iris[sample(nrow(iris)), ]
idx.test = sample(x = seq_len(nrow(df.train)), size = 0.1 * nrow(df.train))

df.test  = df.train[idx.test, ]
df.train = df.train[-idx.test, ]

splits = 3L
breaks = seq(1, nrow(df.train), length.out = splits + 1)

# Save single train sets:
files = character(splits)
for (i in seq_len(splits)) {
    files[i] = paste0("data/iris", i, ".csv")

    temp = df.train[breaks[i]:breaks[i + 1], ]
    write.csv(x = temp, file = files[i], row.names = FALSE)
}

# Save test set:
write.csv(x = df.test, file = "data/iris_test.csv")
```

## Run Gradient Descent

``` r
Rcpp::sourceCpp("src/gradient_descent.cpp")
source("R/grad_descent_lm.R")

myformula = formula(Sepal.Length ~ Petal.Length + Sepal.Width)

mod.lm = lm(myformula, data = df.train)
summary(mod.lm)
## 
## Call:
## lm(formula = myformula, data = df.train)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
## -0.956 -0.232 -0.016  0.218  0.661 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)
## (Intercept)    2.3029     0.2543    9.06  1.5e-15
## Petal.Length   0.4688     0.0175   26.81  < 2e-16
## Sepal.Width    0.5773     0.0714    8.09  3.4e-13
## 
## Residual standard error: 0.325 on 132 degrees of freedom
## Multiple R-squared:  0.847,  Adjusted R-squared:  0.845 
## F-statistic:  367 on 2 and 132 DF,  p-value: <2e-16

mybeta = lmGradientDescent(myformula, data = df.train, iters = 30000L, learning_rate = 0.01,
    mse_eps = 1e-10, trace = FALSE)
```

|              |     lm | grad.descent |     diff |
| ------------ | -----: | -----------: | -------: |
| (Intercept)  | 2.3029 |       2.3006 |   0.0023 |
| Petal.Length | 0.4688 |       0.4689 | \-0.0001 |
| Sepal.Width  | 0.5773 |       0.5780 | \-0.0006 |

### Doing just one step

``` r
actual_beta = coef(mod.lm) * 1.1
mybeta = updateBeta (myformula, data = df.train, learning_rate = 0.01, actual_beta = actual_beta, 
  mse_eps = 1e-10, trace = FALSE, warnings = FALSE)

knitr::kable(data.frame(lm = coef(mod.lm), actual.step = actual_beta, after.step = mybeta$beta))
```

|              |     lm | actual.step | after.step |
| ------------ | -----: | ----------: | ---------: |
| (Intercept)  | 2.3029 |      2.5332 |     2.5216 |
| Petal.Length | 0.4688 |      0.5157 |     0.4699 |
| Sepal.Width  | 0.5773 |      0.6351 |     0.5997 |

## Distributed Linear Model

``` r
source("R/distributed_lm.R")
```

Initialize model:

``` r
files = c("data/iris1.csv", "data/iris2.csv", "data/iris3.csv")
myformula = formula(Sepal.Length ~ Petal.Length + Sepal.Width)

initializeDistributedLinearModel(formula = myformula, out.dir = getwd(), files = files, epochs = 30000L, 
    learning_rate = 0.01, mse_eps = 1e-10, file_reader = read.csv)
```

We can have a look at the created registry:

``` r
load("train_files/registry.rds")
registry
## $file_names
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
## 
## $epochs
## [1] 30000
## 
## $mse_eps
## [1] 1e-10
## 
## $actual_iteration
## [1] 10000
## 
## $formula
## Sepal.Length ~ Petal.Length + Sepal.Width
## 
## $file_reader
## function (file, header = TRUE, sep = ",", quote = "\"", dec = ".", 
##     fill = TRUE, comment.char = "", ...) 
## read.table(file = file, header = header, sep = sep, quote = quote, 
##     dec = dec, fill = fill, comment.char = comment.char, ...)
## <bytecode: 0x5611e0cb45c8>
## <environment: namespace:utils>
## 
## $learning_rate
## [1] 0.01

load("train_files/model.rds")
model
## $mse_average
## [1] 0.1033
## 
## $done
## [1] FALSE
## 
## $beta
## [1] 2.1788 0.4740 0.6116
```

This file is permanent and holds all necessary data to coordinate the
fitting process of the train.

Now we can train a linear model by loading the files specified in files
sequentially:

``` r
trainDistributedLinearModel(regis = "train_files")
## 
## Entering iteration 10000
##  Processing data/iris1.csv
##  Processing data/iris2.csv
##  Processing data/iris3.csv

load("train_files/registry.rds")
registry
## $file_names
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
## 
## $epochs
## [1] 30000
## 
## $mse_eps
## [1] 1e-10
## 
## $actual_iteration
## [1] 10000
## 
## $formula
## Sepal.Length ~ Petal.Length + Sepal.Width
## 
## $file_reader
## function (file, header = TRUE, sep = ",", quote = "\"", dec = ".", 
##     fill = TRUE, comment.char = "", ...) 
## read.table(file = file, header = header, sep = sep, quote = quote, 
##     dec = dec, fill = fill, comment.char = comment.char, ...)
## <bytecode: 0x5611e2621dc8>
## <environment: namespace:utils>
## 
## $learning_rate
## [1] 0.01

load("train_files/model.rds")
model
## $mse_average
## [1] 0.1033
## 
## $done
## [1] FALSE
## 
## $beta
## [1] 2.1788 0.4740 0.6116

max_iters = 10000L

while(registry[["actual_iteration"]] < max_iters) {
    trainDistributedLinearModel(regis = "train_files", silent = TRUE)
    load("train_files/registry.rds")
}

load("train_files/registry.rds")
registry
## $file_names
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
## 
## $epochs
## [1] 30000
## 
## $mse_eps
## [1] 1e-10
## 
## $actual_iteration
## [1] 10000
## 
## $formula
## Sepal.Length ~ Petal.Length + Sepal.Width
## 
## $file_reader
## function (file, header = TRUE, sep = ",", quote = "\"", dec = ".", 
##     fill = TRUE, comment.char = "", ...) 
## read.table(file = file, header = header, sep = sep, quote = quote, 
##     dec = dec, fill = fill, comment.char = comment.char, ...)
## <bytecode: 0x5611e120d390>
## <environment: namespace:utils>
## 
## $learning_rate
## [1] 0.01

load("train_files/model.rds")
model
## $mse_average
## [1] 0.1033
## 
## $done
## [1] FALSE
## 
## $beta
## [1] 2.1788 0.4740 0.6116

knitr::kable(data.frame(lm = coef(mod.lm), actual.step = model[["beta"]]))
```

|              |     lm | actual.step |
| ------------ | -----: | ----------: |
| (Intercept)  | 2.3029 |      2.1788 |
| Petal.Length | 0.4688 |      0.4740 |
| Sepal.Width  | 0.5773 |      0.6116 |
