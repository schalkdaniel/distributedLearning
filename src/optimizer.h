#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// [[depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "models.h"

Rcpp::List gradientDescent (model::Model*, arma::colvec&, double, unsigned int, bool, bool);

# endif // OPTIMIZER_H_