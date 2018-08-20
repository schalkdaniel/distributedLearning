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

RCPP_EXPOSED_CLASS(ModelWrapper);
// RCPP_EXPOSED_CLASS(LinearModelWrapper);

Rcpp::List gradientDescentWrapper (ModelWrapper& mod, arma::colvec& param_start, double learning_rate = 0.01, 
  unsigned int iters_at_once = 1, bool trace = false, bool warnings = false)
{
  return gradientDescent(mod.my_model, param_start, learning_rate, iters_at_once, trace, warnings);
}


// RCPP_EXPOSED_CLASS(GradientDescent);

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
  function("gradientDescent", &gradientDescentWrapper);
}

#endif // MODEL_MODULE_CPP_