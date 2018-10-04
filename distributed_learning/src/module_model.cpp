#ifndef MODEL_MODULE_CPP_
#define MODEL_MODULE_CPP_

#include "models.h"
#include "optimizer.h"

class ModelWrapper
{
public:
  model::Model* my_model;
};

class LinearModelWrapper : ModelWrapper
{
public:
  LinearModelWrapper (arma::mat X, arma::mat y) { my_model = new model::LinearModel (X, y); };

  arma::colvec calculateGradient (arma::colvec& param) const { return my_model->calculateGradient(param); };
  double calculateMSE (arma::colvec& param) const { return my_model->calculateMSE(param); };

  arma::colvec predictNewdata(arma::mat& newdata) const { return my_model->predict(newdata); };
  arma::colvec predict() const { return my_model->predict(); };
};

class LogisticRegressionWrapper : ModelWrapper
{
public:
  LogisticRegressionWrapper (arma::mat X, arma::mat y) { my_model = new model::LogisticRegression (X, y); };

  arma::colvec calculateGradient (arma::colvec& param) const { return my_model->calculateGradient(param); };
  double calculateMSE (arma::colvec& param) const { return my_model->calculateMSE(param); };

  arma::colvec predictNewdata(arma::mat& newdata) const { return my_model->predict(newdata); };
  arma::colvec predict() const { return my_model->predict(); };
};

RCPP_EXPOSED_CLASS(ModelWrapper);

//' Conduct Gradient Descent on a given model
//' 
//' This function applies the Gradient Descent algorithm to a specific model. 
//' 
//' @param model [\code{Model}]\cr
//'   Pointer of the model we want to fit.
//' @param param_start [\code{numeric}]\cr 
//'   The initial parameter.
//' @param learning_rate [\code{numeric(1)}]\cr 
//'   Step size of the gradient updates.
//' @param iters [\code{integer(1)}]\cr 
//'   Number of iterations.
//' @param trace [\code{logical(1)}]\cr 
//'   Flag to specify whether to print the progress or not.
//' @param warnings [\code{logical(1)}]\cr 
//'   Flag to specify whether to print warnings or not.
//' @return [\code{list}] List containing the parameter, the last update, the 
//'   cumulated updates, and the actual MSE of the parameter.
Rcpp::List optGradientDescent (ModelWrapper& mod, arma::colvec& param_start, double learning_rate = 0.01, 
  unsigned int iters = 1, bool trace = false, bool warnings = false)
{
  return optimizer::optGradientDescent(mod.my_model, param_start, learning_rate, iters, trace, warnings);
}

//' Conduct Momentum on a given model
//' 
//' This function applies the Momentum algorithm to a specific model. 
//' 
//' @param model [\code{Model}]\cr
//'   Pointer of the model we want to fit.
//' @param param_start [\code{numeric}]\cr 
//'   The initial parameter.
//' @param learning_rate [\code{numeric(1)}]\cr 
//'   Step size of the gradient updates.
//' @param momentum [\code{numeric(1)}]\cr 
//'   Momentum term, fraction of how much of the previous gradient we add.
//' @param iters [\code{integer(1)}]\cr 
//'   Number of iterations.
//' @param trace [\code{logical(1)}]\cr 
//'   Flag to specify whether to print the progress or not.
//' @param warnings [\code{logical(1)}]\cr 
//'   Flag to specify whether to print warnings or not.
//' @return [\code{list}] List containing the parameter, the last update, the 
//'   cumulated updates, and the actual MSE of the parameter.
Rcpp::List optMomentum (ModelWrapper& mod, arma::colvec& param_start, double learning_rate = 0.01, 
  double momentum = 0.9, unsigned int iters = 1, bool trace = false, bool warnings = false)
{
  return optimizer::optMomentum(mod.my_model, param_start, learning_rate, momentum, iters, trace, warnings);
}

RCPP_MODULE (models)
{
  using namespace Rcpp;

  class_<ModelWrapper>("Model")
  ;

  class_<LinearModelWrapper> ("LinearModel")
    .derives<ModelWrapper> ("Model")

    .constructor<arma::mat, arma::colvec> ("Create the structure for a linear model")
    .method("calculateGradient", &LinearModelWrapper::calculateGradient, "Calculate the gradient w.r.t. a given parameter vector.")
    .method("calculateMSE", &LinearModelWrapper::calculateMSE, "Calculate the MSE w.r.t. a given parameter vector.")
    .method("predictNewdata", &LinearModelWrapper::predictNewdata, "Predict using newdata")
    .method("predict", &LinearModelWrapper::predict, "Predict using the training data")
  ;
  class_<LogisticRegressionWrapper> ("LogisticRegression")
    .derives<ModelWrapper> ("Model")

    .constructor<arma::mat, arma::colvec> ("Create the structure for a logistic regression model")
    .method("calculateGradient", &LogisticRegressionWrapper::calculateGradient, "Calculate the gradient w.r.t. a given parameter vector.")
    .method("calculateMSE", &LogisticRegressionWrapper::calculateMSE, "Calculate the MSE w.r.t. a given parameter vector.")
    .method("predictNewdata", &LogisticRegressionWrapper::predictNewdata, "Predict using newdata")
    .method("predict", &LogisticRegressionWrapper::predict, "Predict using the training data")
  ;
  function("optGradientDescent", &optGradientDescent);
  function("optMomentum", &optMomentum);
}

#endif // MODEL_MODULE_CPP_