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
    return newdata * parameter;
  }
  arma::colvec LinearModel::predict () const
  {
    return X * parameter;
  }

} // namespace model