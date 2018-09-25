#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// [[depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "models.h"

namespace optimizer
{

Rcpp::List gradientDescent (model::Model*, arma::colvec&, double, unsigned int, bool, bool);
Rcpp::List momentum (model::Model*, arma::colvec&, double, double, unsigned int, bool, bool);

} // namespace optimizer

# endif // OPTIMIZER_H_