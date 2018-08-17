// #ifndef GRADIENT_DESCENT_CPP_
// #define GRADIENT_DESCENT_CPP_

// #include <RcppCommon.h>

// #include "models.h"

// namespace Rcpp {

//     // non-intrusive extension via template specialisation
//     template <> Model as(SEXP model);

//     // non-intrusive extension via template specialisation
//     template <> SEXP wrap(const Model& d);
// }

// // [[Rcpp::depends(RcppArmadillo)]]
// #include <RcppArmadillo.h>

// // define template specialisations for as and wrap
// namespace Rcpp {
//     template <> Model as(SEXP modelsexp) {
//         Rcpp::List model(dtsexp);
//         return boost::gregorian::date(dt.getYear(), dt.getMonth(), dt.getDay());
//     }

//     template <> SEXP wrap(const boost::gregorian::date &d) {
//         boost::gregorian::date::ymd_type ymd = d.year_month_day();     // convert to y/m/d struct
//         return Rcpp::wrap( Rcpp::Date( ymd.year, ymd.month, ymd.day ));
//     }
// }

// //' Optimize a model using Gradient Descent
// //' 
// //' This function runs as many gradient descent steps as defined in \code{iters_at_once}.
// //' Note that this function is not able to do something like early stopping in case of 
// //' a non improving MSE or stuff like that. The user is responsible for something like 
// //' that.
// //' 
// //' @param X [\code{arma::mat}]\cr
// //'   Design matrix used within the linear model.
// //' @param y [\code{arma::colvec}]\cr
// //'   Response given as vector.
// //' @param iters [\code{unsigned int}]\cr
// //'   Number of maximal iterations. Could be less if the "epsilon criteria" is hit.
// //' @param learning_rate [\code{double}]\cr
// //'   The step size used for gradient descent. Note: If the mse is not improving the step size
// //'   is shrinked by 20 percent.
// //' @param beta_init [\code{arma::colvec}]\cr
// //'   Initial vector of coefficients used as starting point for the gradient descent.
// //' @param mse_eps [\code{double}]\cr
// //'   Relativ improvement of the MSE. If this boundary is undershot, then the algorithm stops.
// //' @param trace [\code{bool}]\cr
// //'   Flag if the trace should be printed or not.
// //' @param warnings [\code{bool}]\cr
// //'   Flag to specify if warnings should be printed or not.
// //' @return List of parameter vector, the final mse, and a flag if the algorithm was stopped
// //'   by the "epsilon criteria" or after the maximal iterations.
// // [[Rcpp::export]]
// Rcpp::List gradientDescent (arma::colvec& param_start, double learning_rate = 0.01, unsigned int iters_at_once = 1, 
// 	bool trace = false, bool warnings = false, Model model)
// {
// 	// Get Model object from SEXP:
// 	// Rcpp::XPtr<model::Model> temp_model (model);
//   //model::Model instanziated_model = *temp_model;

// 	// Initialize MSE for tracking:
// 	double mse_start = model.calculateMSE(param_start);
// 	double mse_new;

// 	// Initialize vector of update (gradient), a vector of cumulated updates,
// 	// and new parameter vector:
// 	arma::colvec update;
// 	arma::colvec update_cum(param_start.size(), arma::fill::zeros);
// 	arma::colvec param_new = param_start;

// 	// Loop of gradient descent starting in a parameter vector given by the user:
// 	for (unsigned int i = 0; i < iters_at_once; i++) {

// 		// We are dividing by n to minimize the MSE not SSE (this basically shrinks
// 		// the learning rate by 1/n):
// 		update      = model.calculateGradient(param_new);
// 		update_cum += update;
// 		param_new  += learning_rate * update;

// 		mse_new = model.calculateMSE(param_new);

// 	  if ((mse_start - mse_new) < 0 && warnings) {
// 	  	Rcpp::warning("Be careful, MSE is not improving! Reducing the learning rate by 20 percent!");
// 	  	// Be careful, we are changing the value of a pointer here:
// 	  	learning_rate = learning_rate * 0.8;
// 	  }		
// 	  if (trace) {
// 	  	Rcpp::Rcout << "Actual MSE: " << mse_new << std::endl;
// 	  }
// 	  // Set the start MSE to the new one:
// 	  mse_start = mse_new;
// 	}

// 	return Rcpp::List::create(
// 		Rcpp::Named("param_new")   =  param_new, 
// 		Rcpp::Named("udpate")      =  update,
// 		Rcpp::Named("update_cum")  =  update_cum
// 		Rcpp::Named("break_algo")  =  break_algo, 
// 		Rcpp::Named("mse")         =  mse_new
// 	);
// }
// #endif // GRADIENT_DESCENT_CPP_