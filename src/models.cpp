#include "models.h"

namespace model
{

arma::colvec Model::getParameter () const 
{
  return parameter;
}
arma::colvec Model::setParameter (arma::colvec& param)
{
  parameter = param;
}

Model::~Model ()
{
  // std::cout << "Destroy Model" << std::endl;
}


/*
 * Linear Model Implementation
 *
 * In addition to the data matrix X and target variable y we also store X^T * X and X^T * y to speed up computations.
 */
LinearModel::LinearModel (arma::mat X, arma::colvec y) : X(X), y(y)
{
  arma::mat temp = X.t();

  XtX = temp * X;
  Xty = temp * y;   
};

LinearModel::~LinearModel ()
{
  // std::cout << "Destroy Linear Model" << std::endl;
}

arma::colvec LinearModel::calculateGradient (arma::colvec& param) const
{
  return (2 * (Xty - XtX * param) / X.n_rows);
};
double LinearModel::calculateMSE (arma::colvec& param) const
{
  return (arma::accu(arma::pow(y - X * param, 2)) / X.n_rows);
};
arma::colvec LinearModel::predict (arma::mat& newdata) const
{
  // We are just using sigmoidal-link here, 
  return newdata * parameter;
}
arma::colvec LinearModel::predict () const
{
  return X * parameter;
}

/*
 * Logistic Regression Implementation
 *
 */
LogisticRegression::LogisticRegression (arma::mat X, arma::colvec y) : X(X), y(y)
{

};

LogisticRegression::~LogisticRegression ()
{
  // std::cout << "Destroy Logistic Regression Object" << std::endl;
}

arma::colvec LogisticRegression::calculateGradient (arma::colvec& param) const
{
  arma::colvec temp_linear_predictor = X * param;
  arma::colvec temp_sigmoidal = 1 / (1 + arma::exp(- temp_linear_predictor));
  return X.t() * ( (- 2 * y / (1 + arma::exp(- 2 * y % temp_sigmoidal))) % (arma::exp(- temp_linear_predictor) / arma::pow(1 + arma::exp(- temp_linear_predictor), 2)) );
};
double LogisticRegression::calculateMSE (arma::colvec& param) const
{
  return (arma::accu(arma::pow(y - X * param, 2)) / X.n_rows);
};
arma::colvec LogisticRegression::responseFun (arma::colvec& score) const 
{
  return 1 / (1 + arma::exp(-score));
}
arma::colvec LogisticRegression::predict (arma::mat& newdata) const
{
  return responseFun(newdata * parameter);
}
arma::colvec LogisticRegression::predict () const
{
  return responseFun(X * parameter);
}

} // namespace model

