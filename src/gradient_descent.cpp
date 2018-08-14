// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List updateBeta_internal (arma::mat& X, arma::colvec& y, arma::mat& XtX, arma::mat& Xty, 
	arma::colvec& actual_beta, double& actual_mse, double& learning_rate, double& mse_eps, bool& trace)
{
	bool break_algo = false;
	unsigned int n = X.n_rows;

	arma::colvec beta_new = actual_beta + learning_rate * 2 * (Xty - XtX * actual_beta) / n;
	double mse_new = arma::accu(arma::pow(y - X * beta_new, 2)) / n;

	double mse_improvement = (actual_mse - mse_new) / actual_mse;

	if (mse_improvement < 0) {
		Rcpp::warning("Be careful, mse is not improving! Reducing the learning rate by 20 percent!");
		learning_rate = learning_rate * 0.8;
	}		
	if (trace) {
		Rcpp::Rcout << "Actual MSE: " << mse_new << std::endl;
	}
	if (std::abs(mse_improvement) < mse_eps) { 
		Rcpp::Rcout << "Break with an relative MSE improvement of " << mse_improvement << std::endl;
		break_algo = true; 
	}
	return Rcpp::List::create(
		Rcpp::Named("beta") = beta_new, 
		Rcpp::Named("break_algo") = break_algo, 
		Rcpp::Named("mse") = mse_new
	);
}

//' Train a linear model using gradient descent
//' 
//' This function is just a wrapper around the \code{lmGradientDescent()} function written
//' in \code{C++}. 
//' 
//' @param X [\code{arma::mat}]\cr
//'   Design matrix used within the linear model.
//' @param y [\code{arma::colvec}]\cr
//'   Response given as vector.
//' @param iters [\code{unsigned int}]\cr
//'   Number of maximal iterations. Could be less if the "epsilon criteria" is hit.
//' @param learning_rate [\code{double}]\cr
//'   The step size used for gradient descent. Note: If the mse is not improving the step size
//'   is shrinked by 20 percent.
//' @param beta_init [\code{arma::colvec}]\cr
//'   Initial vector of coefficients used as starting point for the gradient descent.
//' @param mse_eps [\code{double}]\cr
//'   Relativ improvement of the MSE. If this boundary is undershot, then the algorithm stops.
//' @param trace [\code{bool}]\cr
//'   Flag if the trace should be printed or not.
//' @returns List of parameter vector, the final mse, and a flag if the algorithm was stopped
//'   by the "epsilon criteria" or after the maximal iterations.
// [[Rcpp::export]]
Rcpp::List lmGradientDescent_internal (arma::mat& X, arma::colvec& y, unsigned int iters = 100, 
  double learning_rate = 0.05, arma::colvec beta_init = 0, double mse_eps = 0.00001, bool trace = false)
{
	arma::colvec beta_hat;
	if (beta_init(0) == 0) {
		beta_hat = arma::randu<arma::colvec>(X.n_cols);
	} else {
		beta_hat = beta_init;
	}

	arma::mat XtX = X.t() * X;
	arma::mat Xty = X.t() * y;

	double mse_new = arma::accu(arma::pow(y - X * beta_hat, 2)) / X.n_rows;

	Rcpp::List steps;

	for (unsigned int i = 0; i < iters; i++) {

		steps = updateBeta_internal(X, y, XtX, Xty, beta_hat, mse_new, learning_rate, mse_eps, trace);

		beta_hat = Rcpp::as<arma::colvec>(steps["beta"]);
		mse_new = Rcpp::as<double>(steps["mse"]);

		if (Rcpp::as<bool>(steps["break_algo"])) { 
			break; 
		}
	}
	return steps;
}