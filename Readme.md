
## Create Test Datasets

To test the functionality we split the `iris` dataset into 3 pieces and
train on these three subsets without sharing data in a distributed
fashion. This is done by averaging gradient descent updates.

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

The `files` object contains the different paths to the datasets which
are required for the fitting process:

``` r
files
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
```

## Run Ordinary Gradient Descent

First of all, load the package using `devtools`:

``` r
devtools::load_all()
## Loading distributedLearning
```

To show what happens on one location we create a `Model` object to
specify the model we want to
fit:

``` r
# "Load" iris dataset and create data matrix. In the final algorithm, this is done by a formula:
X = cbind(Intercept = 1, Petal.Length = df.train[["Petal.Length"]], Sepal.Width = df.train[["Sepal.Width"]])

# Define target variable:
y = df.train[["Sepal.Length"]]

# Now we define a linear model as model for fitting:
lin.mod = LinearModel$new(X, y)

# With that model we can, i.e., calculate the MSE for a given parameter vector:
lin.mod$calculateMSE(rnorm(3))
## [1] 44.42
```

In order to be able to compare our distributed Gradient Descent
algorithm we fit a linear model on the whole dataset:

``` r
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
```

Finally, we run the `gradientDescent()` optimizer for 3000 epochs with a
learning rate of 0.01 by starting at \(\vec{0}\):

``` r
param.start = rep(0, ncol(X))
grad.desc.lm = gradientDescent(lin.mod, param.start, 0.01, 3000, FALSE, FALSE)
str(grad.desc.lm)
## List of 4
##  $ param_new : num [1:3, 1] 1.296 0.511 0.852
##  $ udpate    : num [1:3, 1] 0.00022624 -0.00000954 -0.00006167
##  $ update_cum: num [1:3, 1] 1.296 0.511 0.852
##  $ mse       : num 0.115

knitr::kable(data.frame(lm = coef(mod.lm), grad.descent = grad.desc.lm$param_new, 
  diff = coef(mod.lm) - grad.desc.lm$param_new))
```

|              |     lm | grad.descent |     diff |
| ------------ | -----: | -----------: | -------: |
| (Intercept)  | 2.3029 |       1.2962 |   1.0068 |
| Petal.Length | 0.4688 |       0.5113 | \-0.0425 |
| Sepal.Width  | 0.5773 |       0.8518 | \-0.2744 |

The nice thing about the implementation is that we can pass the last
parameter and continue training for lets say 2000 iterations with a
smaller learning rate:

``` r
param.start.next = grad.desc.lm$param_new
grad.desc.lm.next = gradientDescent(lin.mod, param.start.next, 0.001, 2000, FALSE, FALSE)
str(grad.desc.lm)
## List of 4
##  $ param_new : num [1:3, 1] 1.296 0.511 0.852
##  $ udpate    : num [1:3, 1] 0.00022624 -0.00000954 -0.00006167
##  $ update_cum: num [1:3, 1] 1.296 0.511 0.852
##  $ mse       : num 0.115

knitr::kable(data.frame(lm = coef(mod.lm), grad.descent = grad.desc.lm.next$param_new, 
  diff = coef(mod.lm) - grad.desc.lm.next$param_new))
```

|              |     lm | grad.descent |     diff |
| ------------ | -----: | -----------: | -------: |
| (Intercept)  | 2.3029 |       1.3404 |   0.9625 |
| Petal.Length | 0.4688 |       0.5094 | \-0.0406 |
| Sepal.Width  | 0.5773 |       0.8397 | \-0.2624 |

## Run Distributed Gradient Descent

The technique used is to create a model for each subset and conduct one
or more gradient descent steps. The idea is to save the current state of
the actual model on the disk and reload that setting every time a new
update is available:

1.  On server/location *i* do:
    1.  Read *i*-th dataset
    2.  Load settings or initialize them if not available:
        1.  Define the model and the target variable and formula
        2.  Load latest gradient descent update
    3.  Conduct as many gradient descent steps as defined in advance
2.  Go ahead to server/location *i+1* or make a final average of updates
    if \(i\) was the last one and therefore run again on server/location
    *1*.

### Initialize model

For demonstration we use the three files created above on which we want
to learn a (in this case) Linear Model. First of all, it is necessary to
define all data paths of the data locations:

``` r
files = c("data/iris1.csv", "data/iris2.csv", "data/iris3.csv")
```

Now we initialize the model by setting the formula which is applied on
each of the specified datasets, the model we want to train as
**character** as well as the optimizer, the output directory of the
single steps (each step can be stored if desired), and stuff like the
epochs and learning rate:

``` r
myformula = formula(Sepal.Length ~ Petal.Length + Sepal.Width)

initializeDistributedModel(formula = myformula, model = "LinearModel", optimizer = "gradientDescent", 
  out_dir = getwd(), files = files, epochs = 30000L, learning_rate = 0.01, mse_eps = 1e-10, 
  file_reader = read.csv, overwrite = TRUE)
```

We can have a look at the created registry and the model object:

``` r
load("train_files/registry.rds")
registry
## $file_names
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
## 
## $model
## [1] "LinearModel"
## 
## $optimizer
## [1] "gradientDescent"
## 
## $epochs
## [1] 30000
## 
## $mse_eps
## [1] 1e-10
## 
## $actual_iteration
## [1] 0
## 
## $formula
## Sepal.Length ~ Petal.Length + Sepal.Width
## 
## $file_reader
## function (file, header = TRUE, sep = ",", quote = "\"", dec = ".", 
##     fill = TRUE, comment.char = "", ...) 
## read.table(file = file, header = header, sep = sep, quote = quote, 
##     dec = dec, fill = fill, comment.char = comment.char, ...)
## <bytecode: 0x562f55a18588>
## <environment: namespace:utils>
## 
## $learning_rate
## [1] 0.01
## 
## $save_all
## [1] FALSE

load("train_files/model.rds")
model
## $mse_average
## [1] 0
## 
## $done
## [1] FALSE
```

This file is permanent and holds all necessary data to coordinate the
fitting process of the training.

Now we train a the model by loading the files sequentially. What happens
is that the program looks which files are available on the machine,
grabs the files, and conduct one or more gradient steps. This step is
stored as `.rds` file in the output directory (if `overwrite = TRUE`),
otherwise it is deleted when it is not used anymore.

Finally, we can do an update by calling `trainDistributedModel()`:

``` r
trainDistributedModel(regis = "train_files")
## 
## Entering iteration 0
##  Processing data/iris1.csv
##  Processing data/iris2.csv
##  Processing data/iris3.csv
trainDistributedModel(regis = "train_files")
##   >> Calculate new beta which gives an mse of 3.19279892679591
##   >> Removing train_files/iter0.rds

load("train_files/registry.rds")
registry
## $file_names
## [1] "data/iris1.csv" "data/iris2.csv" "data/iris3.csv"
## 
## $model
## [1] "LinearModel"
## 
## $optimizer
## [1] "gradientDescent"
## 
## $epochs
## [1] 30000
## 
## $mse_eps
## [1] 1e-10
## 
## $actual_iteration
## [1] 1
## 
## $formula
## Sepal.Length ~ Petal.Length + Sepal.Width
## 
## $file_reader
## function (file, header = TRUE, sep = ",", quote = "\"", dec = ".", 
##     fill = TRUE, comment.char = "", ...) 
## read.table(file = file, header = header, sep = sep, quote = quote, 
##     dec = dec, fill = fill, comment.char = comment.char, ...)
## <bytecode: 0x562f573b3028>
## <environment: namespace:utils>
## 
## $learning_rate
## [1] 0.01
## 
## $save_all
## [1] FALSE

load("train_files/model.rds")
model
## $mse_average
## [1] 3.193
## 
## $done
## [1] FALSE
## 
## $beta
## [1] 0.1454 0.8834 0.2628
```

In addition, the training of the model can also be done until the
stopping criteria is hit. This is ether the maximal number of epochs or
the relative improvement of the MSE (specified as `mse_eps` 10^{-10}).
This can be checked by looking at `model[["done"]]`. This returns `TRUE`
if it is done and `FALSE` if not:

``` r
while(! model[["done"]]) {
    trainDistributedModel(regis = "train_files", silent = TRUE)
    load("train_files/model.rds")
}
```

We can have a final look on the estimated parameter by loading the
model:

``` r
load("train_files/registry.rds")
registry[["actual_iteration"]]
## [1] 30001

load("train_files/model.rds")
model
## $mse_average
## [1] 0.1031
## 
## $done
## [1] TRUE
## 
## $beta
## [1] 2.3124 0.4683 0.5752

knitr::kable(data.frame(lm = coef(mod.lm), actual.step = model[["beta"]], diff = coef(mod.lm) - model[["beta"]]))
```

|              |     lm | actual.step |     diff |
| ------------ | -----: | ----------: | -------: |
| (Intercept)  | 2.3029 |      2.3124 | \-0.0095 |
| Petal.Length | 0.4688 |      0.4683 |   0.0006 |
| Sepal.Width  | 0.5773 |      0.5752 |   0.0021 |
