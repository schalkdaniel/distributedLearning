#include "models.h"

// [[Rcpp::export]]
SEXP useLinearModel (arma::mat& X, arma::colvec& y)
{
	Model* temp = new LinearModel(X, y);
	Rcpp::XPtr<Model*> ptr(&temp);

	return Rcpp::wrap(ptr);
}

// [[Rcpp::export]]
double testLinearModel (SEXP& modelptr, arma::colvec& beta)
{
  Rcpp::XPtr<Model*> temp (modelptr);
  return (*temp)->calculateMSE(beta);
}
