#include "optimizer.h"

/**
 * \brief Calculating penalty matrix
 * 
 * This function calculates the penalty matrix for a given number of 
 * parameters (`nparams`) and a given number of differences (`differences`).
 * 
 * \param nparams `unsigned int` Number of params which should be penalized.
 *   This also pretend the number of rows and columns.
 *   
 * \param differences `unsigned int` Number of penalized differences.
 * 
 * \returns `arma::sp_mat` Sparse penalty matrix used for p splines. 
 */

Rcpp::List gradientDescent (model::Model* model, arma::colvec& param_start, double learning_rate = 0.01, 
  unsigned int iters_at_once = 1, bool trace = false, bool warnings = false)
{
  // Get Model object from SEXP:
  // Rcpp::XPtr<model::Model> temp_model (model);
  //model::Model instanziated_model = *temp_model;

  // Initialize MSE for tracking:
  double mse_start = model->calculateMSE(param_start);
  double mse_new;

  // Initialize vector of update (gradient), a vector of cumulated updates,
  // and new parameter vector:
  arma::colvec update;
  arma::colvec update_cum(param_start.size(), arma::fill::zeros);
  arma::colvec param_new = param_start;

  // Loop of gradient descent starting in a parameter vector given by the user:
  for (unsigned int i = 0; i < iters_at_once; i++) {

    // We are dividing by n to minimize the MSE not SSE (this basically shrinks
    // the learning rate by 1/n):
    update      = model->calculateGradient(param_new);
    update_cum += update;
    param_new  += learning_rate * update;

    mse_new = model->calculateMSE(param_new);

    if ((mse_start - mse_new) < 0 && warnings) {
      Rcpp::warning("Be careful, MSE is not improving! Reducing the learning rate by 20 percent!");
      // Be careful, we are changing the value of a pointer here:
      learning_rate = learning_rate * 0.8;
    }   
    if (trace) {
      Rcpp::Rcout << "Actual MSE: " << mse_new << std::endl;
    }
    // Set the start MSE to the new one:
    mse_start = mse_new;
  }

  return Rcpp::List::create(
    Rcpp::Named("param_new")   =  param_new, 
    Rcpp::Named("udpate")      =  update,
    Rcpp::Named("update_cum")  =  update_cum,
    Rcpp::Named("mse")         =  mse_new
  );
}
