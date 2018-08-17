#ifndef MODEL_MODULE_CPP_
#define MODEL_MODULE_CPP_

#include "models.h"

RCPP_EXPOSED_CLASS(Model);
RCPP_EXPOSED_CLASS(LinearModel);


// RCPP_EXPOSED_CLASS(GradientDescent);

RCPP_MODULE (models)
{
  using namespace Rcpp;

  class_<Model>("Model")
  ;

  class_<LinearModel> ("LinearModel")
    .derives<Model> ("Model")

    .constructor<arma::mat, arma::colvec> ("Create the structure for a linear model")
    .method("calculateGradient", &LinearModel::calculateGradient, "Calculate the gradient w.r.t. a given parameter vector.")
    .method("calculateMSE", &LinearModel::calculateMSE, "Calculate the MSE w.r.t. a given parameter vector.")
  ;
}

#endif // MODEL_MODULE_CPP_