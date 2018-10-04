#include "optimizer.h"

namespace optimizer
{
/**
 * \brief Conduct Gradient Descent on a given model
 * 
 * This function applies the Gradient Descent algorithm to a specific model. 
 * 
 * \param model `model::Model*` Pointer of the model we want to fit.
 * \param param_start `arma::colvec` The initial parameter.
 * \param learning_rate `double` Step size of the gradient updates.
 * \param iters `unsigned int` Number of iterations.
 * \param trace `bool` Flag to specify whether to print the progress or not.
 * \param warnings `bool` Flag to specify whether to print warnings or not.
 * \return `Rcpp::List` List containing the parameter, the last update, the 
 *   cumulated updates, and the actual MSE of the parameter. 
 */
Rcpp::List optGradientDescent (model::Model* model, arma::colvec& param_start, double learning_rate = 0.01, 
  unsigned int iters = 1, bool trace = false, bool warnings = false)
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
  for (unsigned int i = 0; i < iters; i++) {

    // We are dividing by n to minimize the MSE not SSE (this basically shrinks
    // the learning rate by 1/n):
    update      = learning_rate * model->calculateGradient(param_new);
    update_cum += update;
    param_new  += update;

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

} // namespace optimizer